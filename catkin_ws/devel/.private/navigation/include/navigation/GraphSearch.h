// Generated by gencpp from file navigation/GraphSearch.msg
// DO NOT EDIT!


#ifndef NAVIGATION_MESSAGE_GRAPHSEARCH_H
#define NAVIGATION_MESSAGE_GRAPHSEARCH_H

#include <ros/service_traits.h>


#include <navigation/GraphSearchRequest.h>
#include <navigation/GraphSearchResponse.h>


namespace navigation
{

struct GraphSearch
{

typedef GraphSearchRequest Request;
typedef GraphSearchResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct GraphSearch
} // namespace navigation


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::navigation::GraphSearch > {
  static const char* value()
  {
    return "09a6e880a7e29d5f1df1f6f7be49541d";
  }

  static const char* value(const ::navigation::GraphSearch&) { return value(); }
};

template<>
struct DataType< ::navigation::GraphSearch > {
  static const char* value()
  {
    return "navigation/GraphSearch";
  }

  static const char* value(const ::navigation::GraphSearch&) { return value(); }
};


// service_traits::MD5Sum< ::navigation::GraphSearchRequest> should match
// service_traits::MD5Sum< ::navigation::GraphSearch >
template<>
struct MD5Sum< ::navigation::GraphSearchRequest>
{
  static const char* value()
  {
    return MD5Sum< ::navigation::GraphSearch >::value();
  }
  static const char* value(const ::navigation::GraphSearchRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::navigation::GraphSearchRequest> should match
// service_traits::DataType< ::navigation::GraphSearch >
template<>
struct DataType< ::navigation::GraphSearchRequest>
{
  static const char* value()
  {
    return DataType< ::navigation::GraphSearch >::value();
  }
  static const char* value(const ::navigation::GraphSearchRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::navigation::GraphSearchResponse> should match
// service_traits::MD5Sum< ::navigation::GraphSearch >
template<>
struct MD5Sum< ::navigation::GraphSearchResponse>
{
  static const char* value()
  {
    return MD5Sum< ::navigation::GraphSearch >::value();
  }
  static const char* value(const ::navigation::GraphSearchResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::navigation::GraphSearchResponse> should match
// service_traits::DataType< ::navigation::GraphSearch >
template<>
struct DataType< ::navigation::GraphSearchResponse>
{
  static const char* value()
  {
    return DataType< ::navigation::GraphSearch >::value();
  }
  static const char* value(const ::navigation::GraphSearchResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // NAVIGATION_MESSAGE_GRAPHSEARCH_H