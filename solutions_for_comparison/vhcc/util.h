#ifndef __UTIL_H__
#define __UTIL_H__


#include <cstring>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cassert>

using namespace std;


#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&)


template<class T>
string toString(const T& data) {
    string result;
    ostringstream converter;
    converter << data;
    result = converter.str();

    return result;
}

inline void ThrowRuntimeError(const string& file_name, const int line, const string& message) {
    throw std::runtime_error("!!!ERROR: " + file_name + ", " + ", line " + toString(line) + ": " + message);
    exit(EXIT_FAILURE);
}
#define THROW_RUNTIME_ERROR(message) ThrowRuntimeError(__FILE__, __LINE__,  message)

#endif // #define __UTIL_H__
