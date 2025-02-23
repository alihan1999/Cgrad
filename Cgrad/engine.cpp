#include <functional>
#include <unordered_set>
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include "engine.h"


Value::Value(float data, std::unordered_set<std::shared_ptr<Value>> _children, std::string op) {
    this->data = data;
    this->grad = 0.0;
    this->_children = std::move(_children);
    this->op = std::move(op);
    this->_backward = [this] {
        for (auto& child : this->_children) {
            child->_backward();
        }
    };
}


float Value::get_data() {
    return data;
}

void Value::set_data(float data) {
    this->data = data;
}

float Value::get_grad() const {
    return grad;
}
void Value::set_grad(float val) {
    this->grad=val;
}

std::unordered_set<std::shared_ptr<Value>> Value::get_children() {
    return _children;
}


// += operator
void operator +=(std::shared_ptr<Value>&x,const std::shared_ptr<Value> &y){
    x = x+y;
}
void operator +=(std::shared_ptr<Value>&x,float y){
    x = x+y;
}


//--------------------------------------------------------------//
// summation / + operator 
std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {

    auto out = std::make_shared<Value>(data + other->data, std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other}, "+");

    out->_backward = [this,other,out] {
        grad += out->grad;
        other->grad += out->grad;

    };
    return out;
}


std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& x, const std::shared_ptr<Value>& y) {
    return (*x) + y;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& x, float y) {
    return (*x) + y;
}

std::shared_ptr<Value> Value::operator+(float f) {
    std::shared_ptr<Value> rhs = std::make_shared<Value>(f);
    return (*shared_from_this())+rhs;
}
std::shared_ptr<Value> operator +(float f,const std::shared_ptr<Value>& rhs){
    std::shared_ptr<Value> other = std::make_shared<Value>(f);
    return (*other)+rhs;
}

//------------------------------------//

//multiplication / * operator

std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {

    auto out = std::make_shared<Value>(data * other->data, std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other}, "*");

    out->_backward = [this,other,out] {
        grad += other->data * out->grad;
        other->grad += data * out->grad;
        
    };
    return out;
}
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& x, const std::shared_ptr<Value>& y) {
    return (*x) * y;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& x, float y) {
    return (*x) * y;
}
std::shared_ptr<Value> operator *(float f,const std::shared_ptr<Value>& rhs){
    std::shared_ptr<Value> lhs = std::make_shared<Value>(f);
    return (*lhs)*rhs;
}

std::shared_ptr<Value> Value::operator*(float f){

    std::shared_ptr<Value> other = std::make_shared<Value>(f);
    return (*shared_from_this())*other;
}


//-------------------------------------------//

//subtract / - operator

std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {
    return shared_from_this() + (other->operator-());
}
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& x, const std::shared_ptr<Value>& y) {
    return (*x) - y;
}
std::shared_ptr<Value> Value::operator-(float f){
    std::shared_ptr<Value> rhs = std::make_shared<Value>(f);
    return (*shared_from_this())-rhs;
}
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& x, float y) {
    return (*x) - y;
}
std::shared_ptr<Value> operator -(float f,const std::shared_ptr<Value>& rhs){
    std::shared_ptr<Value> lhs = std::make_shared<Value>(f);
    return (*lhs)-rhs;
}

std::shared_ptr<Value> Value ::operator-(){
    return shared_from_this()*std::make_shared<Value>(-1.0);
}

//-------------------------------------//



//power ------------/
std::shared_ptr<Value> Value::pow(const std::shared_ptr<Value>& other) {

    auto out = std::make_shared<Value>(std::pow(data,other->data),std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other}, "^");

    out->_backward = [this,other,out] {
        grad += other->data*std::pow(data,other->data-1)*out->grad;
        other->grad += std::pow(data,other->data)*std::log(data)*out->grad;
        
    };
    return out;
}
std::shared_ptr<Value> pow(const std::shared_ptr<Value>& x, const std::shared_ptr<Value>& y) {
    return (*x).pow(y);
}

std::shared_ptr<Value> pow(const std::shared_ptr<Value>& x, float y) {
    return (*x).pow(y);
}

std::shared_ptr<Value> Value::pow(float f){

    std::shared_ptr<Value> rhs = std::make_shared<Value>(f);
    return (*shared_from_this()).pow(rhs);
}

//-------------------------------------------

//division / /operator
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& x, const std::shared_ptr<Value>& y) {
    return (*x) * y->pow(-1);
}

std::shared_ptr<Value> Value:: operator /(const std::shared_ptr<Value>& other){

return (*shared_from_this())*other->pow(-1);
};


std::shared_ptr<Value> Value::operator/(float f) {
    std::shared_ptr<Value> other = std::make_shared<Value>(f);
    return (*shared_from_this())*other->pow(-1);
}
std::shared_ptr<Value> operator /(float f,const std::shared_ptr<Value>& rhs){
    std::shared_ptr<Value> lhs = std::make_shared<Value>(f);
    return (*lhs)/rhs;
}
std::shared_ptr<Value> operator /(const std::shared_ptr<Value>& x, float y) {
    return (*x)/y;
}


bool Value:: operator < (const std::shared_ptr<Value>& other){
    return data < other->data;
}

bool operator < (const std::shared_ptr<Value> &x,const std::shared_ptr<Value>& y ){
    return (*x)<y;
}


// exponential

std::shared_ptr<Value> Value::exp(){
    std::shared_ptr<Value> out = std::make_shared<Value>(std::exp(data));
    out->_backward = [this,out](){
        grad += out->data*out->grad;
    };
    return out;
}

//tanh
std::shared_ptr<Value> Value:: tanh(){
    return (exp()-1/exp())/(exp()+1/exp());
}

//sigmoid
std::shared_ptr<Value> Value:: sigmoid(){
    return 1/(1+1/exp());
}


std::ostream & operator << (std::ostream&os,const std::shared_ptr<Value>&other){
    os << other->data;
    return os;
}


//--------------------------------------------//
void Value::backward() {
    std::vector<std::shared_ptr<Value>> nodes;
    std::unordered_set<std::shared_ptr<Value>> vis;

    std::function<void(const std::shared_ptr<Value>&)> topo_sort = [&](const std::shared_ptr<Value>& v) {
        if (!vis.count(v)) {
            vis.insert(v);

            for (const auto& child : v->_children) {
                topo_sort(child);
            }
            nodes.push_back(v);
        }
    };

    topo_sort(shared_from_this());

    grad = 1.0;

    std::reverse(nodes.begin(),nodes.end());

    for(const auto &x:nodes){
        x->_backward();
    }
}

void Value::zero_grad(){

    std::unordered_set<std::shared_ptr<Value>> vis;
    std::vector<std::shared_ptr<Value>> nodes;
    std::function<void(const std::shared_ptr<Value>&)> _zero_grad = [&](const std::shared_ptr<Value> & v){
        if(!vis.count(v)){
            vis.insert(v);
            nodes.push_back(v);
            for(auto & c:v->_children)_zero_grad(c);
        }
    };
    _zero_grad(shared_from_this());
    for(auto &c:nodes)c->grad=0.0;
}
