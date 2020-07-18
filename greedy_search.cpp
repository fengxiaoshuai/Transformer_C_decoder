// greedy_search.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <iomanip>
#include <mkl.h>
#include <memory>
#include <vector>
#include <assert.h>
#include <numeric>
#include <string>
#include <algorithm>
#include <time.h>
#include <fstream>
#include <chrono>
#include "decode.h"
using namespace std;
using namespace NMT;

void load(vector<float>& weight, string dir)
{
	ifstream input(dir);

	if (input.fail())
	{
		cout << "File does not exist" << endl;
		cout << "Exit program" << endl;
		return;
	}
	float num=0.0;
	while (input>>num)  // 当没有读到文件结尾
	{
		weight.push_back(num);
		//cout << num << endl;
	}
	input.close();

}

void load_layer_weight(vector<vector<float>>& layer_weight, int num)
{
	cout << "start read layer " << num << " weight" << endl;
	vector<float> layer_self_scale;//0
	vector<float> layer_self_bias;//1
	vector<float> layer_self_q;//2
	vector<float> layer_self_k;//3
	vector<float> layer_self_v;//4
	vector<float> layer_self_last;//5

	vector<float> layer_encdec_scale;//6
	vector<float> layer_encdec_bias;//7
	vector<float> layer_encdec_q;//8
	vector<float> layer_encdec_k;//cache k//9
	vector<float> layer_encdec_v;//cache v//10
	vector<float> layer_encdec_last;//11

	vector<float> layer_ffn_scale;//12
	vector<float> layer_ffn_bias;//13
	vector<float> layer_ffn_first_weight;//14
	vector<float> layer_ffn_first_bias;//15
	vector<float> layer_ffn_second_weight;//16
	vector<float> layer_ffn_second_bias;//17
	
	vector<float> layer_self_position_key;//18
	vector<float> layer_self_position_value;//19
 

	cout << "...:load self attention weight" << endl;
	string name = "./weight/layer_" + to_string(num) ;
	load(layer_self_scale, name + "_self_scale.txt");
	load(layer_self_bias, name + "_self_bias.txt");
	load(layer_self_q, name + "_self_q.txt");
	load(layer_self_k, name + "_self_k.txt");
	load(layer_self_v, name + "_self_v.txt");
	load(layer_self_last, name + "_self_last.txt");
	load(layer_self_position_key, name + "_self_position_key.txt");
	load(layer_self_position_value, name + "_self_position_value.txt");
	cout << "...:load encdec attention weight" << endl;
	load(layer_encdec_scale, name + "_encdec_scale.txt");
	load(layer_encdec_bias, name + "_encdec_bias.txt");
	load(layer_encdec_q, name + "_encdec_q.txt");
	load(layer_encdec_k, name + "_encdec_k.txt");
	load(layer_encdec_v, name + "_encdec_v.txt");
	load(layer_encdec_last, name + "_encdec_last.txt");
	cout << "...:load read fnn weight" << endl;
	load(layer_ffn_scale, name + "_ffn_scale.txt");
	load(layer_ffn_bias, name + "_ffn_bias.txt");
	load(layer_ffn_first_weight, name + "_ffn_first_weight.txt");
	load(layer_ffn_first_bias, name + "_ffn_first_bias.txt");
	load(layer_ffn_second_weight, name + "_ffn_second_weight.txt");
	load(layer_ffn_second_bias, name + "_ffn_second_bias.txt");


	layer_weight.push_back(layer_self_scale);
	layer_weight.push_back(layer_self_bias);
	layer_weight.push_back(layer_self_q);
	layer_weight.push_back(layer_self_k);
	layer_weight.push_back(layer_self_v);
	layer_weight.push_back(layer_self_last);

	layer_weight.push_back(layer_encdec_scale);
	layer_weight.push_back(layer_encdec_bias);
	layer_weight.push_back(layer_encdec_q);
	layer_weight.push_back(layer_encdec_k);
	layer_weight.push_back(layer_encdec_v);
	layer_weight.push_back(layer_encdec_last);

	layer_weight.push_back(layer_ffn_scale);
	layer_weight.push_back(layer_ffn_bias);
	layer_weight.push_back(layer_ffn_first_weight);
	layer_weight.push_back(layer_ffn_first_bias);
	layer_weight.push_back(layer_ffn_second_weight);
	layer_weight.push_back(layer_ffn_second_bias);

	layer_weight.push_back(layer_self_position_key);
	layer_weight.push_back(layer_self_position_value);
	
	cout << "...:end layer " << num << " weight" << endl;
}


int main()
{
        size_t head = 16;
        size_t hidden = 1024;
        size_t layer = 6;
        size_t decode_length = 18;
        size_t vocab_num = 32768;
        size_t ffn = 4096;
	size_t batch_size = 2;
	size_t length = 8;
	vector<int> target_language = {1,1};
        vector<int> mask = {1,1,1,1,1,0,0,0,   1,1,1,1,1,1,0,0};
	

	vector<float> encode_out;
	load(encode_out, "./weight/encode_out.txt");

	cout << "start load embedding" << endl;
	vector<float> weight_embedding;
	load(weight_embedding, "./weight/embedding.txt");
	vector<float> language_embedding;
	load(language_embedding, "./weight/language_embedding.txt");
	cout << "end load embedding" << endl;

	vector<float> weight_scale;
	load(weight_scale, "./weight/scale.txt");

	vector<float> weight_bias;
	load(weight_bias, "./weight/bias.txt");

	cout << "start load logits" << endl;
	vector<float> logit_weight;
	load(logit_weight, "./weight/logit.txt");
	cout << "end load logits" << endl;

	vector<vector<vector<float>>> weight(layer);
	for(int i = 0; i<layer; i++)
	{
		load_layer_weight(weight[i], i);
	}
	Decoder decoder = Decoder(
				head,
				hidden,
				layer,
				decode_length,
				vocab_num,
				ffn,
				weight,
				weight_embedding,
				language_embedding,
				weight_scale,
				weight_bias,
				logit_weight);
	cout<<encode_out.size()<<"--this encode_out"<<endl;
	auto time1 = chrono::steady_clock::now();
	decoder.Translate(encode_out, batch_size, length, target_language,  mask);
	auto time2 = chrono::steady_clock::now();
        cout << "decode time:" << (chrono::duration_cast<chrono::duration<double>>(time2 - time1)).count() << endl;
	return 0;
}


