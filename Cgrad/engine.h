#ifndef ENGINE_H
#define ENGINE_H

#include <functional>
#include <unordered_set>
#include <string>
#include <memory>


class Value : public std::enable_shared_from_this<Value> {
private:
    float data;
    float grad;
    std::unordered_set<std::shared_ptr<Value>> _children;
    std::function<void()> _backward;  
    std::string op;

public:
    Value(float data, std::unordered_set<std::shared_ptr<Value>>_children={}, std::string op="");

    void set_grad(float );
    float get_data();
    void set_data(float );
    float get_grad() const;

    void backward();
    void zero_grad();

    std::unordered_set<std::shared_ptr<Value>>  get_children();

    std::shared_ptr<Value> exp();

    //Activations
    std::shared_ptr<Value> tanh();
    std::shared_ptr<Value> sigmoid();


    friend void operator += (std::shared_ptr<Value>& ,const std::shared_ptr<Value>&);
    friend void operator += (std::shared_ptr<Value>& ,float);

    bool operator < (const std::shared_ptr<Value>&);
    friend bool operator < (const std::shared_ptr<Value> & ,const std::shared_ptr<Value> & );


    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& );
    std::shared_ptr<Value> operator+(float ) ;
    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& , const std::shared_ptr<Value>& );
    friend std::shared_ptr<Value> operator+(float , const std::shared_ptr<Value>& );
    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>&,float );

    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& );
    std::shared_ptr<Value> operator*(float ) ;
    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& , const std::shared_ptr<Value>& );
    friend std::shared_ptr<Value> operator*(float , const std::shared_ptr<Value>& );
    friend std::shared_ptr<Value> operator *(const std::shared_ptr<Value>&,float);

    std::shared_ptr<Value> operator-();
    
    std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& );
    std::shared_ptr<Value> operator-(float ) ;
    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& , const std::shared_ptr<Value>& );
    friend std::shared_ptr<Value> operator-(float , const std::shared_ptr<Value>& );
    friend std::shared_ptr<Value> operator -(const std::shared_ptr<Value>&,float);


    std::shared_ptr<Value> pow(const std::shared_ptr<Value>& );
    std::shared_ptr<Value> pow(float ) ;
    friend std::shared_ptr<Value> pow(const std::shared_ptr<Value>& , const std::shared_ptr<Value>& );
    friend std::shared_ptr<Value> pow(const std::shared_ptr<Value>& ,float);

    std::shared_ptr<Value> operator /(const std::shared_ptr<Value>& );
    std::shared_ptr<Value> operator  /(float ) ;
    friend std::shared_ptr<Value> operator  /(const std::shared_ptr<Value>& , const std::shared_ptr<Value>& );
    friend std::shared_ptr<Value> operator /(float , const std::shared_ptr<Value>& );
    friend std::shared_ptr<Value> operator /(const std::shared_ptr<Value>&,float);

    friend std::ostream&  operator << (std::ostream&,const std::shared_ptr<Value>&);



};


#endif  