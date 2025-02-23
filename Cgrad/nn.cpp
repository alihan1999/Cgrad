#include<bits/stdc++.h>
#include "nn.h"
#include<memory>
#include<algorithm>

void Module::zero_grad(){
    for(auto & w:parameters()){
        w->set_grad(0.0);
    }
}

Neuron::Neuron(int n){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0,1.0);

    for(int i=0;i<n;++i)w.push_back(std::make_shared<Value>(dist(gen)));
    b = std::make_shared<Value>(dist(gen));
    };



std::shared_ptr<Value> Neuron:: operator () ( std::vector<std::shared_ptr<Value>> x){
    auto out = std::make_shared<Value>(0.0);
    std::vector<std::shared_ptr<Value>> mul;
    std::transform(w.begin(),w.end(),x.begin(),std::back_inserter(mul),[](const std::shared_ptr<Value> & aa,const std::shared_ptr<Value> bb){return aa*bb;});
    for(const auto & item:mul){
        out+=item;
    }
    out+=b;
  
    return out;
    
};

std::vector<std::shared_ptr<Value>> Neuron::parameters(){
    std::vector<std::shared_ptr<Value>> params;
    for(auto p:w)params.push_back(p);
    params.push_back(b);
    return params;

}

Layer::Layer(int in,int out):input_dim(in),output_dim(out){
    for(int i=0;i<out;++i)neurons.push_back(std::make_shared<Neuron>(in));
};




std::vector<std::shared_ptr<Value>> Layer::operator()(std::vector<std::shared_ptr<Value>> x){
    std::vector<std::shared_ptr<Value>> out;
    for(auto n:neurons){
       out.push_back((*n)(x));
    }
    return out;
}

std::vector<std::shared_ptr<Value>> Layer::parameters(){
    std::vector<std::shared_ptr<Value>> params;
    for(auto n:neurons){
        for(auto p:n->parameters())params.push_back(p);
    }
    return params;

}

Linear::Linear(int in, std::vector<int> output_dims):input_dim(in){
    
    this->output_dims.resize(output_dims.size());
    this->output_dims = std::move(output_dims);
  
    layers.push_back(std::make_shared<Layer>(input_dim,this->output_dims[0]));
    int i=0;
    while(i+1!=this->output_dims.size())layers.push_back(std::make_shared<Layer>(this->output_dims[i],this->output_dims[i+1])),i++;
}
std::vector<std::shared_ptr<Value>> Linear::operator()(std::vector<std::shared_ptr<Value>> x){
    for(auto l:layers){
        x=(*l)(x);
    }
    return x;
}
std::vector<std::shared_ptr<Value>> Linear::parameters(){
    std::vector<std::shared_ptr<Value>> params;
    for(auto l:layers){
        for(auto p:l->parameters())params.push_back(p);
    }
    return params;

}
