// Copyright (c) 2012, 2019 Scott Niekum, Joshua Whitley
// All rights reserved.
//
// Software License Agreement (BSD License 2.0)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of {copyright_holder} nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <pluginlib/class_loader.hpp>

#include <string>
#include <map>
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "ml_classifiers/zero_classifier.hpp"
#include "ml_classifiers/nearest_neighbor_classifier.hpp"
#include "ml_classifiers/srv/add_class_data.hpp"
#include "ml_classifiers/srv/classify_data.hpp"
#include "ml_classifiers/srv/clear_classifier.hpp"
#include "ml_classifiers/srv/create_classifier.hpp"
#include "ml_classifiers/srv/load_classifier.hpp"
#include "ml_classifiers/srv/save_classifier.hpp"
#include "ml_classifiers/srv/train_classifier.hpp"

using namespace ml_classifiers;  // NOLINT
using std::string;
using std::cout;
using std::endl;

std::map<string, std::shared_ptr<Classifier>> classifier_list;
pluginlib::ClassLoader<Classifier> c_loader("ml_classifiers", "ml_classifiers::Classifier");
std::shared_ptr<rclcpp::Node> cl_srv_node;

bool createHelper(string class_type, std::shared_ptr<Classifier> & c)
{
  try {
    c = std::shared_ptr<Classifier>(c_loader.createUnmanagedInstance(class_type));
  } catch (pluginlib::PluginlibException & ex) {
    RCLCPP_ERROR(
      cl_srv_node->get_logger(),
      "Classifer plugin failed to load! Error: %s",
      ex.what());
  }

  return true;
}

bool createCallback(
  const std::shared_ptr<rmw_request_id_t> req_hdr,
  const std::shared_ptr<ml_classifiers::srv::CreateClassifier::Request> req,
  std::shared_ptr<ml_classifiers::srv::CreateClassifier::Response> res)
{
  string id = req->identifier;
  std::shared_ptr<Classifier> c;

  if (!createHelper(req->class_type, c)) {
    res->success = false;
    return false;
  }

  if (classifier_list.find(id) != classifier_list.end()) {
    RCLCPP_INFO(
      cl_srv_node->get_logger(),
      "WARNING: ID already exists, overwriting: %s",
      req->identifier.c_str());
    classifier_list.erase(id);
  }

  classifier_list[id] = c;

  res->success = true;
  return true;
}

bool addCallback(
  const std::shared_ptr<rmw_request_id_t> req_hdr,
  const std::shared_ptr<ml_classifiers::srv::AddClassData::Request> req,
  std::shared_ptr<ml_classifiers::srv::AddClassData::Response> res)
{
  string id = req->identifier;

  if (classifier_list.find(id) == classifier_list.end()) {
    res->success = false;
    return false;
  }

  for (size_t i = 0; i < req->data.size(); i++) {
    classifier_list[id]->addTrainingPoint(req->data[i].target_class, req->data[i].point);
  }

  res->success = true;
  return true;
}

bool trainCallback(
  const std::shared_ptr<rmw_request_id_t> req_hdr,
  const std::shared_ptr<ml_classifiers::srv::TrainClassifier::Request> req,
  std::shared_ptr<ml_classifiers::srv::TrainClassifier::Response> res)
{
  string id = req->identifier;

  if (classifier_list.find(id) == classifier_list.end()) {
    res->success = false;
    return false;
  }

  RCLCPP_INFO(
    cl_srv_node->get_logger(),
    "Training %s",
    id.c_str());

  classifier_list[id]->train();
  res->success = true;
  return true;
}

bool clearCallback(
  const std::shared_ptr<rmw_request_id_t> req_hdr,
  const std::shared_ptr<ml_classifiers::srv::ClearClassifier::Request> req,
  std::shared_ptr<ml_classifiers::srv::ClearClassifier::Response> res)
{
  string id = req->identifier;

  if (classifier_list.find(id) == classifier_list.end()) {
    res->success = false;
    return false;
  }

  classifier_list[id]->clear();
  res->success = true;
  return true;
}

bool saveCallback(
  const std::shared_ptr<rmw_request_id_t> req_hdr,
  const std::shared_ptr<ml_classifiers::srv::SaveClassifier::Request> req,
  std::shared_ptr<ml_classifiers::srv::SaveClassifier::Response> res)
{
  string id = req->identifier;

  if (classifier_list.find(id) == classifier_list.end()) {
    res->success = false;
    return false;
  }

  classifier_list[id]->save(req->filename);
  res->success = true;
  return true;
}

bool loadCallback(
  const std::shared_ptr<rmw_request_id_t> req_hdr,
  const std::shared_ptr<ml_classifiers::srv::LoadClassifier::Request> req,
  std::shared_ptr<ml_classifiers::srv::LoadClassifier::Response> res)
{
  string id = req->identifier;

  std::shared_ptr<Classifier> c;

  if (!createHelper(req->class_type, c)) {
    res->success = false;
    return false;
  }

  if (!c->load(req->filename)) {
    res->success = false;
    return false;
  }

  if (classifier_list.find(id) != classifier_list.end()) {
    RCLCPP_WARN(
      cl_srv_node->get_logger(),
      "WARNING: ID already exists, overwriting: %s",
      req->identifier.c_str());
    classifier_list.erase(id);
  }
  classifier_list[id] = c;

  res->success = true;
  return true;
}

bool classifyCallback(
  const std::shared_ptr<rmw_request_id_t> req_hdr,
  const std::shared_ptr<ml_classifiers::srv::ClassifyData::Request> req,
  std::shared_ptr<ml_classifiers::srv::ClassifyData::Response> res)
{
  string id = req->identifier;

  for (size_t i = 0; i < req->data.size(); i++) {
    string class_num = classifier_list[id]->classifyPoint(req->data[i].point);
    res->classifications.push_back(class_num);
  }

  return true;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  cl_srv_node = rclcpp::Node::make_shared("classifier_server");

  cl_srv_node->create_service<ml_classifiers::srv::CreateClassifier>(
    "create_classifier", createCallback);
  cl_srv_node->create_service<ml_classifiers::srv::AddClassData>(
    "add_class_data", addCallback);
  cl_srv_node->create_service<ml_classifiers::srv::TrainClassifier>(
    "train_classifier", trainCallback);
  cl_srv_node->create_service<ml_classifiers::srv::ClearClassifier>(
    "clear_classifier", clearCallback);
  cl_srv_node->create_service<ml_classifiers::srv::SaveClassifier>(
    "save_classifier", saveCallback);
  cl_srv_node->create_service<ml_classifiers::srv::LoadClassifier>(
    "load_classifier", loadCallback);
  cl_srv_node->create_service<ml_classifiers::srv::ClassifyData>(
    "classify_data", classifyCallback);

  RCLCPP_INFO(cl_srv_node->get_logger(), "Classifier services now ready");
  rclcpp::spin(cl_srv_node);

  return 0;
}
