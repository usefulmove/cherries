#include "tracker.h"
#include "tracker_node.h"
#include "helper.h"
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/int64.hpp"
#include <functional>


#include <QApplication>

// #include <rclcpp/executors/multi_threaded_executor.hpp>

class ProjectorApp : public QApplication
{
public:
  // rclcpp::Node::SharedPtr nh_;

  explicit ProjectorApp(int &argc, char **argv)
      : QApplication(argc, argv)
  {
  }

  ~ProjectorApp()
  {
  }

  void SetEncoder(long count)
  {
    // RCLCPP_INFO(rclcpp::get_logger("app"), "current: '%ld'", count);
    try
    {
      if (start)
        widget_ptr->SetEncoder(count);
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(rclcpp::get_logger("app"), "set encoder error: %s", e.what());
    }
  }

  void SetImage(QImage projector, QImage conveyor, long reference_count)
  {
    // RCLCPP_INFO(rclcpp::get_logger("app"), "current: '%ld'", count);
    try
    {
      // RCLCPP_ERROR(rclcpp::get_logger("app"), "frame callback with  encoder count: %ld", reference_count);
      if (start)
        widget_ptr->SetImage(projector, conveyor, reference_count);
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(rclcpp::get_logger("app"), "set image error: %s", e.what());
    }
  }

  int exec(int screen_number)
  {
    // nh_.show();

    // to do behave nicely if this screen does not exist.
    RCLCPP_INFO(rclcpp::get_logger("app"), "strart on scrren : %d",  screen_number);

    auto callback = std::bind(&ProjectorApp::SetScreen, this, std::placeholders::_1);

    Helper helper;
    Tracker widget(&helper, callback);

    // make sure the screen asked for exists and if not use the primary screen
    int screen_index = 0;  //default
    if (QGuiApplication::screens().length() > screen_number)
      screen_index = screen_number;

    widget_ptr = &widget;
    QScreen *screen = QGuiApplication::screens()[screen_index]; //default to

    widget.move(screen->geometry().x(), screen->geometry().y());
    widget.resize(screen->geometry().width(), screen->geometry().height());
    widget.showFullScreen();

    start = true;

    return QApplication::exec();
  }

  void SetScreen(int screen_number)
  {
    // nh_.show();
    RCLCPP_INFO(rclcpp::get_logger("main"), "getting scren handle");

   int screen_index = 0;  //default
    if (QGuiApplication::screens().length() > screen_number)
      screen_index = screen_number;

    QScreen *screen = QGuiApplication::screens()[screen_index]; // specify which screen to use
    widget_ptr->SetScreen(screen);
  }

private:
  // delcare objects for the class

  Tracker *widget_ptr;
  bool start = false;
};
// // SIGINT handler function
// void handle_shutdown(int s)
// {
//     ROS_INFO("Shutting down node...");
//     ros::shutdown(); // Step 1: stopping ROS event loop
//     QCoreApplication::exit(0); // Step 2: stopping Qt event loop
// }

// std::shared_ptr<ProjectorApp> app;

bool spin_ = true;
void stop_spin(int s)
{
  rclcpp::shutdown();
  // app->quit();
  QCoreApplication::quit();
  RCLCPP_INFO(rclcpp::get_logger("main"), "quit called");
}

void spin_node(std::shared_ptr<TrackerNode> node)
{
  RCLCPP_INFO(rclcpp::get_logger("spin_thread"), "starting thread");
  rclcpp::executors::SingleThreadedExecutor executor;

  // executor.add_node(minpub);
  executor.add_node(node);

  // while (spin_)
  // {
  //   executor.spin_once();
  // }

  executor.spin();

  RCLCPP_INFO(rclcpp::get_logger("spin_thread"), "ending thread");

  return;
};

int main(int argc, char *argv[])
{
  try
  {
    RCLCPP_INFO(rclcpp::get_logger("main"), "rclcpp init");
    rclcpp::init(argc, argv);

    RCLCPP_INFO(rclcpp::get_logger("main"), "make node");
    std::shared_ptr<TrackerNode> node = std::make_shared<TrackerNode>();

    RCLCPP_INFO(rclcpp::get_logger("main"), "set sigint handler");
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = stop_spin;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);

    RCLCPP_INFO(rclcpp::get_logger("main"), "creae app instance");
    // // Instantiating Qt application object
    ProjectorApp app(argc, argv);
    //   struct sigaction sigIntHandler2;
    // sigIntHandler2.sa_handler = QCoreApplication::quit;
    // sigemptyset(&sigIntHandler2.sa_mask);
    // sigIntHandler2.sa_flags = 0;

    // auto thing = ;
    node->SetEncoderCallback(std::bind(&ProjectorApp::SetEncoder, &app, std::placeholders::_1));
    node->SetFrameCallback(std::bind(&ProjectorApp::SetImage, &app, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    node->SetScreenCallback(std::bind(&ProjectorApp::SetScreen, &app, std::placeholders::_1));

    // sigIntHandler.sa_flags = 0;

    auto spinFuture = std::async(spin_node, node);

    // sigaction(SIGINT, &sigIntHandler, NULL);

    // Exit code: from Qt
    RCLCPP_INFO(rclcpp::get_logger("main"), "run app");
    return app.exec(node->GetScreen()); // 
    //return app.exec(1); // node->GetScreen()
    // to do make this start on the correct screen

    RCLCPP_INFO(rclcpp::get_logger("main"), "app exited");

    spinFuture.wait();
    RCLCPP_INFO(rclcpp::get_logger("main"), "shutting down");
  }
  catch (const std::exception &e)
  {
    qInfo() << e.what();
  }
}

// int main(int argc, char** args)
// {

//     // Instantiating Qt application object
//     ProjectorApp app(argc, args);

//     rclcpp::MultiThreadedExecutor executor = rclcpp::MultiThreadedExecutor();
//     executor.add_node(nh_);

//     # Start the ROS2 node on a separate thread
//     std::thread nh_thread = std::thread(&(executor.spin))
//     nh_thread.start()

//     # Let the app running on the main thread
//     try{
//         int rval = app.exec();
//     } final {

//     }

//     finally:
//         hmi_node.get_logger().info("Shutting down ROS2 Node . . .")
//         hmi_node.destroy_node()
//         executor.shutdown()
// }