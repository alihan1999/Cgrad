#ifndef NN_H
#define NN_H    

#include"engine.h"
#include<memory>

class Module{
    public:
    void zero_grad();
    virtual std::vector<std::shared_ptr<Value>> parameters() =0;
};

class Neuron:public Module{

    private:
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;

    public:
    Neuron(int);
    std::shared_ptr<Value> operator() ( std::vector<std::shared_ptr<Value>> );
    std::vector<std::shared_ptr<Value>> parameters();


};

class Layer:public Module{
    private:
    std::vector<std::shared_ptr<Neuron>> neurons;
    int input_dim,output_dim;

    public:
    Layer(int,int);

    std::vector<std::shared_ptr<Value>> operator () (std::vector<std::shared_ptr<Value>>);
    std::vector<std::shared_ptr<Value>> parameters();



};

class Linear:public Module{
    private:
    std::vector<std::shared_ptr<Layer>> layers;
    int input_dim;
    std::vector<int> output_dims;

    public:
    Linear(int, std::vector<int>);
    std::vector<std::shared_ptr<Value>> operator () (std::vector<std::shared_ptr<Value>> );
    std::vector<std::shared_ptr<Value>> parameters();




};
#endif