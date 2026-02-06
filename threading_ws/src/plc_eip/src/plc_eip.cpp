// Copyright 2016 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cipster_api.h>
#include <condition_variable> // std::condition_variable, std::cv_status

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int64.hpp"
#include "rclcpp/qos.hpp"
using namespace std::chrono_literals;
#include "cherry_interfaces/msg/encoder_count.hpp"
#include "cherry_interfaces/srv/set_lights.hpp"
#include "cherry_interfaces/srv/reset_latches.hpp"
#include "cherry_interfaces/srv/latch_robot.hpp"
#include "cherry_interfaces/msg/inputs.hpp"
#include "cherry_interfaces/msg/outputs.hpp"
#include "cherry_interfaces/msg/hsc.hpp"

/*******************************************************************************
 * Copyright (c) 2012, Rockwell Automation, Inc.
 * All rights reserved.
 *
 ******************************************************************************/

#include <cipster_api.h>
#include <string.h>
#include <stdlib.h>
#include <bitset>
#include <thread>

#define DEMO_APP_INPUT_ASSEMBLY_NUM 100                 // 0x064
#define DEMO_APP_OUTPUT_ASSEMBLY_NUM 150                // 0x096
#define DEMO_APP_CONFIG_ASSEMBLY_NUM 151                // 0x097
#define DEMO_APP_HEARTBEAT_INPUT_ONLY_ASSEMBLY_NUM 152  // 0x098
#define DEMO_APP_HEARTBEAT_LISTEN_ONLY_ASSEMBLY_NUM 153 // 0x099
#define DEMO_APP_EXPLICT_ASSEMBLY_NUM 154               // 0x09A

// global variables for demo application (4 assembly data fields)  ***********

uint8_t g_assembly_data064[128]; // Input
uint8_t g_assembly_data096[128]; // Output
uint8_t g_assembly_data097[64];  // Config
uint8_t g_assembly_data09A[128]; // Explicit

EipStatus ApplicationInitialization()
{
  // create 3 assembly object instances
  // INPUT
  CreateAssemblyInstance(DEMO_APP_INPUT_ASSEMBLY_NUM,
                         ByteBuf(g_assembly_data064, sizeof(g_assembly_data064)));

  // OUTPUT
  CreateAssemblyInstance(DEMO_APP_OUTPUT_ASSEMBLY_NUM,
                         ByteBuf(g_assembly_data096, sizeof(g_assembly_data096)));

  // CONFIG
  CreateAssemblyInstance(DEMO_APP_CONFIG_ASSEMBLY_NUM,
                         ByteBuf(g_assembly_data097, sizeof(g_assembly_data097)));

  // Heart-beat output assembly for Input only connections
  CreateAssemblyInstance(DEMO_APP_HEARTBEAT_INPUT_ONLY_ASSEMBLY_NUM,
                         ByteBuf(0, 0));

  // Heart-beat output assembly for Listen only connections
  CreateAssemblyInstance(DEMO_APP_HEARTBEAT_LISTEN_ONLY_ASSEMBLY_NUM,
                         ByteBuf(0, 0));

  // assembly for explicit messaging
  CreateAssemblyInstance(DEMO_APP_EXPLICT_ASSEMBLY_NUM,
                         ByteBuf(g_assembly_data09A, sizeof(g_assembly_data09A)));

  // Reserve some connection instances for the above assemblies:

  ConfigureExclusiveOwnerConnectionPoint(
      DEMO_APP_OUTPUT_ASSEMBLY_NUM,
      DEMO_APP_INPUT_ASSEMBLY_NUM,
      DEMO_APP_CONFIG_ASSEMBLY_NUM);

  // Reserve a connection instance that can connect without a config_path
  ConfigureExclusiveOwnerConnectionPoint(
      DEMO_APP_OUTPUT_ASSEMBLY_NUM,
      DEMO_APP_INPUT_ASSEMBLY_NUM,
      -1); // config path may be omitted

  ConfigureInputOnlyConnectionPoint(
      DEMO_APP_HEARTBEAT_INPUT_ONLY_ASSEMBLY_NUM,
      DEMO_APP_INPUT_ASSEMBLY_NUM,
      DEMO_APP_CONFIG_ASSEMBLY_NUM);

  ConfigureListenOnlyConnectionPoint(
      DEMO_APP_HEARTBEAT_LISTEN_ONLY_ASSEMBLY_NUM,
      DEMO_APP_INPUT_ASSEMBLY_NUM,
      DEMO_APP_CONFIG_ASSEMBLY_NUM);

  return kEipStatusOk;
}

void HandleApplication()
{
  // check if application needs to trigger a connection
}

void NotifyIoConnectionEvent(CipConn *aConn, IoConnectionEvent io_connection_event)
{
  // maintain a correct output state according to the connection state
  int consuming_id = aConn->ConsumingPath().GetInstanceOrConnPt();
  int producing_id = aConn->ProducingPath().GetInstanceOrConnPt();

  // silence unused variables
  (void)consuming_id;
  (void)producing_id;
  (void)io_connection_event;
}

bool BeforeAssemblyDataSend(AssemblyInstance *instance)
{
  // update data to be sent e.g., read inputs of the device
  /*In this sample app we mirror the data from out to inputs on data receive
   * therefore we need nothing to do here. Just return true to inform that
   * the data is new.
   */

  if (instance->Id() == DEMO_APP_EXPLICT_ASSEMBLY_NUM)
  {
    /* do something interesting with the existing data
     * for the explicit get-data-attribute message */
  }

  return true;
}

EipStatus ResetDevice()
{
  // add reset code here
  return kEipStatusOk;
}

EipStatus ResetDeviceToInitialConfiguration(bool also_reset_comm_params)
{
  // reset the parameters

  // then perform device reset
  // silence unused variables
  (void)also_reset_comm_params;

  return kEipStatusOk;
}

void RunIdleChanged(uint32_t run_idle_value)
{
  (void)run_idle_value;
}

#include <stdlib.h>
EipStatus ApplicationInitialization(); // my business.

long sync_time = 0;

// encoder
long encoder_count = 0;
long stored_count = 0;
long encoder_latches[6] = {};
long robot_count = 0;

// light latches
int8_t top_latch_byte;
int8_t bot_latch_byte;
bool top_light_latches[6] = {};
bool bot_light_latches[6] = {};

// variable for synchronizing exectution
std::condition_variable cv_start_acquisition_ack;
std::condition_variable cv_set_top_light_ack;
std::condition_variable cv_set_bot_light_ack;
std::condition_variable cv_set_light_ack;
std::condition_variable cv_robot_latch_ack;

// status bits
bool start_acquisition_ack;
bool acquisition_done;
bool top_light_status = 0;
bool bot_light_status = 0;
bool set_top_light_ack;
bool set_bot_light_ack;
bool trigger_ready;
bool latch_robot_signal_ack;

// control bits
bool start_acquisition;
bool acquisition_done_ack;
bool set_top_light;
bool set_top_light_value;
bool set_bot_light;
bool set_bot_light_value;
bool latch_robot_signal;

// io status dints
uint inputs;
uint outputs;
uint hsc;

int trigger_id_out;
int trigger_id_in;

class PlcPublisher : public rclcpp::Node
{
public:
  PlcPublisher()
      : Node("plc_publisher")
  {
    rclcpp::QoS publisherQoS = rclcpp::SensorDataQoS();
    publisherQoS.deadline(1000us);
    publisherQoS.lifespan(1000us);

    publisher_encoder_ = this->create_publisher<cherry_interfaces::msg::EncoderCount>("encoder", publisherQoS);
    publisher_inputs_ = this->create_publisher<cherry_interfaces::msg::Inputs>("inputs", publisherQoS);
    publisher_outputs_ = this->create_publisher<cherry_interfaces::msg::Outputs>("outputs", publisherQoS);
    publisher_hsc_ = this->create_publisher<cherry_interfaces::msg::HSC>("hsc", publisherQoS);
    timer_ = this->create_wall_timer(
        100us, std::bind(&PlcPublisher::timer_callback, this));
  }

  long last_sync = -1;
  uint8_t in_data[128] = {};

  // int last_value;
  void timer_callback()
  {
    try
    {

      if (last_sync != sync_time)
      {
        auto message = cherry_interfaces::msg::EncoderCount();

        // long value;
        // std::memcpy(&value, g_assembly_data064, sizeof(long));

        // RCLCPP_INFO(this->get_logger(), "Publishing: '%ld'", message.data);
        // if (last_value != value) {
        message.count = encoder_count;
        message.mm = (int)(mm_per_count * encoder_count);
        message.count_stored = stored_count;
        message.mm_stored = (int)(mm_per_count * stored_count);
        publisher_encoder_->publish(message);
        //  last_value = value;
        //}

        publisher_inputs_->publish(setInputMsg(inputs));
        publisher_outputs_->publish(setOutputMsg(outputs));
        publisher_hsc_->publish(setHscMsg(hsc));

        last_sync = sync_time;
      }
    }
    catch (const std::exception &e)
    {
      RCLCPP_INFO(this->get_logger(), "Error publishing eip data");
    }
  }

  bool getbit(uint int_val, int index)
  {
    return ((int_val & (1 << index)) > 0);
  }

  cherry_interfaces::msg::Inputs setInputMsg(uint input_value)
  {
    auto message = cherry_interfaces::msg::Inputs();
    message.trigger_ready = getbit(input_value, 0);
    message.in1 = getbit(input_value, 1);
    message.in2 = getbit(input_value, 2);
    message.in3 = getbit(input_value, 3);
    message.in4 = getbit(input_value, 4);
    message.in5 = getbit(input_value, 5);
    message.in6 = getbit(input_value, 6);
    message.in7 = getbit(input_value, 7);
    message.in8 = getbit(input_value, 8);
    message.in9 = getbit(input_value, 9);
    message.in10 = getbit(input_value, 10);
    message.in11 = getbit(input_value, 11);
    message.in12 = getbit(input_value, 12);
    message.in13 = getbit(input_value, 13);
    message.in14 = getbit(input_value, 14);
    message.in15 = getbit(input_value, 15);
    message.in16 = getbit(input_value, 16);
    message.in17 = getbit(input_value, 17);
    message.in18 = getbit(input_value, 18);
    message.in19 = getbit(input_value, 19);
    message.in20 = getbit(input_value, 20);
    message.in21 = getbit(input_value, 21);
    message.in22 = getbit(input_value, 22);
    message.in23 = getbit(input_value, 23);
    message.in24 = getbit(input_value, 24);
    message.in25 = getbit(input_value, 25);
    message.in26 = getbit(input_value, 26);
    message.in27 = getbit(input_value, 27);
    message.in28 = getbit(input_value, 28);
    message.in29 = getbit(input_value, 29);
    message.in30 = getbit(input_value, 30);
    message.in31 = getbit(input_value, 31);

    return message;
  }

  cherry_interfaces::msg::Outputs setOutputMsg(uint input_value)
  {
    auto message = cherry_interfaces::msg::Outputs();
    message.top_light = getbit(input_value, 0);
    message.bot_light = getbit(input_value, 1);
    message.robot_latch = getbit(input_value, 2);
    message.out3 = getbit(input_value, 3);
    message.out4 = getbit(input_value, 4);
    message.out5 = getbit(input_value, 5);
    message.out6 = getbit(input_value, 6);
    message.out7 = getbit(input_value, 7);
    message.out8 = getbit(input_value, 8);
    message.out9 = getbit(input_value, 9);
    message.out10 = getbit(input_value, 10);
    message.out11 = getbit(input_value, 11);
    message.out12 = getbit(input_value, 12);
    message.out13 = getbit(input_value, 13);
    message.out14 = getbit(input_value, 14);
    message.out15 = getbit(input_value, 15);
    message.out16 = getbit(input_value, 16);
    message.out17 = getbit(input_value, 17);
    message.out18 = getbit(input_value, 18);
    message.out19 = getbit(input_value, 19);
    message.out20 = getbit(input_value, 20);
    message.out21 = getbit(input_value, 21);
    message.out22 = getbit(input_value, 22);
    message.out23 = getbit(input_value, 23);
    message.out24 = getbit(input_value, 24);
    message.out25 = getbit(input_value, 25);
    message.out26 = getbit(input_value, 26);
    message.out27 = getbit(input_value, 27);
    message.out28 = getbit(input_value, 28);
    message.out29 = getbit(input_value, 29);
    message.out30 = getbit(input_value, 30);
    message.out31 = getbit(input_value, 31);

    return message;
  }

  cherry_interfaces::msg::HSC setHscMsg(uint input_value)
  {
    auto message = cherry_interfaces::msg::HSC();
    message.en = getbit(input_value, 0);
    message.soft_preset = getbit(input_value, 1);
    message.gen_error = getbit(input_value, 2);
    message.a0 = getbit(input_value, 3);
    message.b0 = getbit(input_value, 4);
    message.z0 = getbit(input_value, 5);

    return message;
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<cherry_interfaces::msg::EncoderCount>::SharedPtr publisher_encoder_;
  rclcpp::Publisher<cherry_interfaces::msg::Inputs>::SharedPtr publisher_inputs_;
  rclcpp::Publisher<cherry_interfaces::msg::Outputs>::SharedPtr publisher_outputs_;
  rclcpp::Publisher<cherry_interfaces::msg::HSC>::SharedPtr publisher_hsc_;

  double mm_per_count = 0.0877377650089162;
};

class PlcServices : public rclcpp::Node
{
public:
  PlcServices()
      : Node("plc_services")
  {
    service_setlights_ = this->create_service<cherry_interfaces::srv::SetLights>("set_lights", std::bind(&PlcServices::set_lights, this, std::placeholders::_1, std::placeholders::_2));
    service_resetLatches_ = this->create_service < cherry_interfaces::srv::ResetLatches>(
                                                       "reset_latches",
                                                       std::bind(
                                                           &PlcServices::reset_latches, this,
                                                           std::placeholders::_1,
                                                           std::placeholders::_2));
    service_latchRobot_ = this->create_service<cherry_interfaces::srv::LatchRobot>(
      "latch_robot",
      std::bind(
        &PlcServices::robot_latch, this,
        std::placeholders::_1,
        std::placeholders::_2));
  }

  void set_lights(const std::shared_ptr<cherry_interfaces::srv::SetLights::Request> request,
                  std::shared_ptr<cherry_interfaces::srv::SetLights::Response> response)
  {
    set_top_light = true;
    set_bot_light = true;
    set_top_light_value = request->top_light;
    set_bot_light_value = request->bot_light;

    std::mutex mtx_light;
    std::unique_lock<std::mutex> lck_light(mtx_light);
    response->status = 0;
    if (cv_set_light_ack.wait_for(lck_light, std::chrono::milliseconds(500)) == std::cv_status::timeout)
    {
      response->status = -1;
    }

    response->top_light = top_light_status;
    response->bot_light = bot_light_status;
    set_top_light = false;
    set_bot_light = false;
  }

  void reset_latches(const std::shared_ptr<cherry_interfaces::srv::ResetLatches::Request> request,
                     std::shared_ptr<cherry_interfaces::srv::ResetLatches::Response> response)
  {
    trigger_id_out = request->frame_id;
    set_top_light_value = request->top_light;
    set_bot_light_value = request->bot_light;
    start_acquisition = true;
    set_top_light = true;
    set_bot_light = true;

    std::mutex mtx;
    std::unique_lock<std::mutex> lck(mtx);
    if (cv_start_acquisition_ack.wait_for(lck, std::chrono::milliseconds(500)) == std::cv_status::timeout)
    {
      response->status = -1;
    }
    else
    {
      response->status = 0;
    }

    {
      response->frame_id = trigger_id_out;
      response->top_light = top_light_status;
      response->bot_light = bot_light_status;

      start_acquisition = false;
      set_top_light = false;
      set_bot_light = false;
    }
  }

  void robot_latch(const std::shared_ptr<cherry_interfaces::srv::LatchRobot::Request> request,
                     std::shared_ptr<cherry_interfaces::srv::LatchRobot::Response> response)
  {
    try {
      latch_robot_signal = true;

      std::mutex mtx;
      std::unique_lock<std::mutex> lck(mtx);
      if (cv_robot_latch_ack.wait_for(lck, std::chrono::milliseconds(5000)) == std::cv_status::timeout)
      {
        response->encoder_count = 0;
        response->encoder_mm = 0;
        response->status = -1;
        RCLCPP_INFO(this->get_logger(), "Timeout latching robot encoder count.");

      }
      else
      {
        response->encoder_count = encoder_count;
        response->encoder_mm = (int)(mm_per_count * encoder_count);
        response->status = 2;
      }
    } catch (const std::exception &e)
    {
      RCLCPP_INFO(this->get_logger(), "Error latching robot encoder count.");
    } 
    
    latch_robot_signal = false;


    
  }


  rclcpp::Service<cherry_interfaces::srv::SetLights>::SharedPtr service_setlights_;
  rclcpp::Service<cherry_interfaces::srv::ResetLatches>::SharedPtr service_resetLatches_;
  rclcpp::Service<cherry_interfaces::srv::LatchRobot>::SharedPtr service_latchRobot_;
  double mm_per_count = 0.0877377650089162;
};

bool spin = true;
void my_handler(int s)
{
  (void)s;
  spin = false;
}

bool parse_mac(const char *mac_str, uint8_t mac_out[6])
{
  uint b[6];

  if (6 == sscanf(mac_str, "%x:%x:%x:%x:%x:%x", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) ||
      6 == sscanf(mac_str, "%x-%x-%x-%x-%x-%x", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]))
  {
    for (int i = 0; i < 6; ++i)
      mac_out[i] = b[i];

    return true;
  }

  return false;
}

std::shared_ptr<PlcPublisher> minpub;

void setSync()
{
  auto time = std::chrono::system_clock::now(); // get the current time
  auto since_epoch = time.time_since_epoch();   // get the duration since epoch

  // I don't know what system_clock returns
  // I think it's uint64_t nanoseconds since epoch
  // Either way this duration_cast will do the right thing
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(since_epoch);
  sync_time = micros.count(); // just like java (new Date()).getTime();
}

uint status_word = 0;
uint ctrl_word = 0;
std::bitset<32> status_bits;

bool getbit(uint int_val, int index)
{
  return ((int_val & (1 << index)) > 0);
}

EipStatus AfterAssemblyDataReceived(AssemblyInstance *aInstance,
                                    OpMode aMode, int aBytesReceivedCount)
{
  EipStatus status = kEipStatusOk;

  (void)aMode;
  (void)aBytesReceivedCount;

  // handle the data received e.g., update outputs of the device
  switch (aInstance->Id())
  {
  case DEMO_APP_OUTPUT_ASSEMBLY_NUM:
    // Data for the output assembly has been received.
    // Mirror it to the inputs
    // minpub->timer_callback();
    std::memcpy(
        &status_word,
        g_assembly_data096,
        sizeof(status_word));

    std::memcpy(
        &encoder_count,
        g_assembly_data096 + 4,
        sizeof(encoder_count));

    std::memcpy(
        &encoder_latches,
        g_assembly_data096 + 12,
        sizeof(encoder_latches));

    std::memcpy(
        &trigger_id_in,
        g_assembly_data096 + 60,
        sizeof(trigger_id_in));

    std::memcpy(
        &inputs,
        g_assembly_data096 + 64,
        sizeof(inputs));
    std::memcpy(
        &outputs,
        g_assembly_data096 + 68,
        sizeof(outputs));
    std::memcpy(
        &hsc,
        g_assembly_data096 + 72,
        sizeof(hsc));
    std::memcpy(
        &top_latch_byte,
        g_assembly_data096 + 76,
        sizeof(top_latch_byte));
    std::memcpy(
        &bot_latch_byte,
        g_assembly_data096 + 77,
        sizeof(bot_latch_byte));
    std::memcpy(
        &stored_count,
        g_assembly_data096 + 80,
        sizeof(stored_count));
        
    std::memcpy(
        &robot_count,
        g_assembly_data096 + 88,
        sizeof(robot_count));

    // status bits
    start_acquisition_ack = getbit(status_word, 0);
    acquisition_done = getbit(status_word, 1);
    set_top_light_ack = getbit(status_word, 8);
    top_light_status = getbit(status_word, 9);
    set_bot_light_ack = getbit(status_word, 10);
    bot_light_status = getbit(status_word, 11);
    latch_robot_signal_ack = getbit(status_word, 5);

    if (start_acquisition_ack && set_bot_light_ack && set_top_light_ack)
      cv_start_acquisition_ack.notify_one();
    if (set_bot_light_ack)
      cv_set_bot_light_ack.notify_one();
    if (set_top_light_ack)
      cv_set_top_light_ack.notify_one();
    if (set_bot_light_ack && set_top_light_ack)
    {
      cv_set_light_ack.notify_one();
    }

    if (latch_robot_signal_ack)
    {
      cv_robot_latch_ack.notify_one();
    }
    

    // control bits
    ctrl_word = 0;
    ctrl_word += ((int)start_acquisition) << 0;
    ctrl_word += ((int)acquisition_done_ack) << 1;
    ctrl_word += ((int)set_top_light) << 8;
    ctrl_word += ((int)set_top_light_value) << 9;
    ctrl_word += ((int)set_bot_light) << 10;
    ctrl_word += ((int)set_bot_light_value) << 11;
    ctrl_word += ((int)latch_robot_signal) << 5;

    memcpy(&g_assembly_data064[0], &ctrl_word,
           sizeof(ctrl_word));

    memcpy(&g_assembly_data064[4], &trigger_id_out,
           sizeof(trigger_id_out));

    // memcpy(&in_data[0], &g_assembly_data096[0],
    //        sizeof(in_data));

    setSync();
    break;

  case DEMO_APP_EXPLICT_ASSEMBLY_NUM:
    // do something interesting with the new data from
    // the explicit set-data-attribute message
    break;

  case DEMO_APP_CONFIG_ASSEMBLY_NUM:
    /* Add here code to handle configuration data and check if it is ok
     * The demo application does not handle config data.
     * However in order to pass the test we accept any data given.
     * EIP_ERROR
     */
    status = kEipStatusOk;
    break;
  }

  return status;
}

void spin_eip()
{
  // handle eip comms
  while (spin)
  {
    if (kEipStatusOk != NetworkHandlerProcessOnce())
    {
      break;
    }
  }
}

void spin_services()
{
  rclcpp::executors::SingleThreadedExecutor executor;

  // minpub = std::make_shared<PlcPublisher>();
  std::shared_ptr<PlcServices> plcio = std::make_shared<PlcServices>();

  // executor.add_node(minpub);
  executor.add_node(plcio);

  while (spin)
  {
    executor.spin_once();
  }
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  // rclcpp::spin(std::make_shared<PlcPublisher>());

  rclcpp::executors::SingleThreadedExecutor executor;

  minpub = std::make_shared<PlcPublisher>();
  // std::shared_ptr<PlcServices> plcio = std::make_shared<PlcServices>();

  executor.add_node(minpub);
  // executor.add_node(plcio);

  struct sigaction sigIntHandler;

  sigIntHandler.sa_handler = my_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;

  sigaction(SIGINT, &sigIntHandler, NULL);

  // settings for ethernet ip comms
  const char *ip_addr = "10.0.0.10";
  const char *ip_mask = "255.0.0.0";
  const char *ip_gateway = "10.0.0.1";
  const char *domain = "test.com";
  const char *host_addr = "cherryvision";
  const char *mac_addr = "00:30:64:59:F4:91";

  // EipStatus ApplicationInitialization(); // my business.
  //  ApplicationInitialization();  // my business.

  // unique_connection_id should be sufficiently random or incremented
  // and stored in non-volatile memory each time the device boots.
  uint16_t unique_connection_id = rand();

  // Setup the CIP stack early, before calling any stack Configuration functions.
  CipStackInit(unique_connection_id);

  // fetch Internet address info from the platform
  ConfigureNetworkInterface(ip_addr, ip_mask, ip_gateway);
  ConfigureDomainName(domain);
  ConfigureHostName(host_addr);

  uint8_t my_mac_address[6];
  if (!parse_mac(mac_addr, my_mac_address))
  {
    throw std::invalid_argument("Bad macaddress format. It can use either : or - to separate:\n"
                                " e.g. 00:15:C5:BF:D0:87 or 00-15-C5-BF-D0-87");
  }

  ConfigureMacAddress(my_mac_address);

  // for a real device the serial number should be unique per device
  SetDeviceSerialNumber(1);

  if (ApplicationInitialization() != kEipStatusOk)
  {
    // TODO change this to a more appropriate throe type
    throw std::invalid_argument("Unable to initialize Assembly instances\n");
  }

  // Setup Network only after Configure*() calls above
  if (NetworkHandlerInitialize() != kEipStatusOk)
  {
    // TODO change this to a more appropriate throe type
    throw std::invalid_argument("Unable to initialize NetworkHandlers\n");
  }

  std::thread t1(spin_eip);
  std::thread t2(spin_services);

  while (spin)
  {

    // moved to a different thread
    // // handle eip comms
    // if (kEipStatusOk != NetworkHandlerProcessOnce())
    // {
    //   break;
    // }

    // handle ROS2 node
    executor.spin_once();
  }

  printf("\ncleaning up and ending...\n");

  // clean up network state
  NetworkHandlerFinish();

  // close remaining sessions and connections, cleanup used data
  ShutdownCipStack();

  rclcpp::shutdown();
  return 0;
}
