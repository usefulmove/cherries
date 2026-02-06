#include "tracker.h"
#include <math.h>
#include <QDebug>
#include <rclcpp/rclcpp.hpp>

#include <QPainter>
#include <QPaintEvent>
#include <QWidget>
#include <QPaintEvent>
// using Alloc = std::pmr::polymorphic_allocator<void>;
// #include "./ui_tracker.h"

Tracker::Tracker(Helper *helper, std::function<void(int number)> Screen_cb, QWidget *parent)
    : QOpenGLWidget(parent),
      helper(helper),
      elapsed(0)
{
  update_timer_ = new QTimer(this);
  update_timer_->setInterval(16); // ask to update the screen every 16 ms or 60fps
  update_timer_->start();

  // subscription_encoder_->set_on_new_qos_event_callback(&Tracker::deadlineCallback, RCL_SUBSCRIPTION_REQUESTED_DEADLINE_MISSED);

  connect(update_timer_, SIGNAL(timeout()), this, SLOT(onUpdate()));

  setMinimumSize(500, 500);
  setAutoFillBackground(false);

  x_ = 1.920;
  y_ = 1.3;
  scaling_factor_ = 891.0;
  helper->scaling = scaling_factor_;
  show_grid_ = false;
  rotation_ = 0;
  rotation_matrix_ = {1.0, 0.0, 1.0, 0.0};

  screen_ = 0;
  screen_changed_callback_ = Screen_cb;
}

void Tracker::SetEncoder(long count)
{

  RCLCPP_DEBUG(rclcpp::get_logger("tracker"), "Encoder mm: '%ld'", count);

  encoderCount_ = count;

  mm_ = count;
  pixels_ = mm_to_pixels(count);
  // RCLCPP_INFO(rclcpp::get_logger("tracker"), "current: '%ld'", encoderCount_);
}

void Tracker::SetImage(QImage projector, QImage conveyor, long reference_count)
{
  std::lock_guard<std::mutex> guard(image_mtx_);
  conveyorImage_ = QPixmap::fromImage(conveyor);
  projectorImage_ = QPixmap::fromImage(projector);
  referenceMm_ = reference_count;
  referencePixels_ = mm_to_pixels(reference_count);

  RCLCPP_DEBUG(rclcpp::get_logger("tracker"), "images updated with count : '%ld'", referenceMm_);
}

Tracker::~Tracker()
{

  qInfo("done waiting");

  delete update_timer_;
}


void Tracker::SetScreen(QScreen *screen){
  screen_ = screen;
  change_screen_ = true;
}

// only call this from gui thread
void Tracker::set_screen(){
      try
    {
      RCLCPP_DEBUG(rclcpp::get_logger("tracker"), "Move screen!");
      this->showNormal();
      this->move(screen_->geometry().x(), screen_->geometry().y());
      this->resize(screen_->geometry().width(), screen_->geometry().height());
      this->showFullScreen();
    }
    catch (const std::exception &e)
    {
      // RCLCPP_ERROR(
      //     nh_->get_logger(), "Error setting screen! : %s",
      //     e.what());
    }

    change_screen_ = false;
}

void Tracker::animate()
{
  // elapsed = (elapsed + qobject_cast<QTimer*>(sender())->interval()) % 1000;

  if(change_screen_)
    set_screen();

  update();
}

void Tracker::onUpdate()
{
  // if (!rclcpp::ok())
  // {
  //   close();
  //   return;
  // }

  // rclcpp::spin_some(nh_);

  // updateTurtles();
  // update();
  animate();
}

void Tracker::paintEvent(QPaintEvent *event)
{
  if (sizeof(cherries_translated_) == 0)
  {
    return;
  }

  QPainter painter;
  painter.begin(this);
  painter.setRenderHint(QPainter::Antialiasing);
  std::lock_guard<std::mutex> guard(image_mtx_);
  helper->paint(&painter, event, projectorImage_, conveyorImage_, mm_ - referenceMm_, pixels_ - referencePixels_, encoderCount_);

  // if (show_grid_)
  // {
  //   helper->paint_grid(&painter, event);
  // }

  // QRect rectangle(0,0,1920,1080);
  // painter.fillRect(event->rect(), Qt::red);

  painter.end();
}

Cherry_cpp Tracker::translate(Cherry_cpp cherry_conveyor)
{
  // this is  rotation transformation using
  // using [ x ] = [ cos(theta)  -sin(theta) ] * [ x' ]
  //       [ y ]   [ sin(theta)   cos(theta) ]   [ y' ]
  // we can get the point locations in a 180 deg
  // rotated coordinate frame

  // tracnlastion
  double x_trans = (cherry_conveyor.X - x_);
  double y_trans = (cherry_conveyor.Y - y_);

  // rotate & translate the cherries
  double x_meters = rotation_matrix_[0] * x_trans + rotation_matrix_[1] * y_trans;
  double y_meters = rotation_matrix_[2] * x_trans + rotation_matrix_[3] * y_trans;

  // scale to pixels
  double x_pels = x_meters * scaling_factor_;
  double y_pels = y_meters * scaling_factor_;

  // translate into pixels
  Cherry_cpp cherry_projector(x_pels, y_pels, cherry_conveyor.Type);

  return cherry_projector;
}

void Tracker::CherryArrayChanged(cherry_interfaces::msg::CherryArray cherries)
{
  // update the internal array of cherries
  // i suppose I don;t need this
  // cherries_ = cherries;
  qInfo("received frame");

  // // translate the cherry values
  // std::vector<Cherry_cpp> cherries_translated = translate_msg(cherries);
  // Frame frame = Frame(cherries_translated, pixels(cherries.encoder_count));
  // conveyor.Add(frame);

  // try
  // {
  //   qInfo() << "start thread";
  //   drawThread = std::thread(&Tracker::addFrame, this, cherries);

  //   drawThread.join();
  // }
  // catch (const std::exception &e)
  // {
  //   qInfo() << e.what();
  // }

  // int nextEncoderCount = cherries.encoder_count;
  // QPixmap nextPixMap = conveyor.getPixmap(encoderCount_);

  // // update the internal variable
  // // since this is also used by th animiate command, we use a mutex to prevent wierd
  // // behavior when both threads try to access the object at the same time.
  // cherry_mtx_.lock();
  // conveyorImage_ = nextPixMap;
  // referenceCount_ = nextEncoderCount;
  // cherry_mtx_.unlock();
}

void Tracker::addFrame(cherry_interfaces::msg::CherryArray cherries)
{
  // translate the cherry values
  try
  {
    std::vector<Cherry_cpp> cherries_translated = translate_msg(cherries);
    int pixels = mm_to_pixels(cherries.encoder_count);
    Frame frame = Frame(cherries_translated, cherries.encoder_count);
    conveyor.Add(frame);
  }
  catch (const std::exception &e)
  {
    qInfo() << e.what();
  }
}

void Tracker::EncoderCountChanged(std_msgs::msg::Int64 msg)
{

  encoderCount_ = msg.data;
  pixels_ = mm_to_pixels(msg.data);
  hist.add(encoderCount_);
  qInfo() << "encoder act: " << encoderCount_;
  missed = 0;
}

void Tracker::deadlineCallback(rclcpp::QOSDeadlineRequestedInfo info)
{

  // esetimate what the encoder count should be
  missed++;

  if (missed > 15)
    return;

  encoderCount_ = hist.predict(missed);
  qInfo() << "encoder est: " << encoderCount_ << info.total_count << info.total_count_change;
}

std::vector<Cherry_cpp> Tracker::translate_msg(cherry_interfaces::msg::CherryArray msg)
{
  std::vector<Cherry_cpp> cherry_array = {};
  // cherry_interfaces::msg::CherryArray cherries_conveyor = msg.cherries;
  // RCLCPP_INFO(nh_->get_logger(), "Cherry vector length: '%ld'", cherries_conveyor.cherries.size());
  for (unsigned int argi = 0; argi < msg.cherries.size(); argi++)
  {
    Cherry_cpp cherry_conveyor(
        msg.cherries[argi].x * 1000,
        msg.cherries[argi].y * 1000,
        msg.cherries[argi].type);

    cherry_array.push_back(cherry_conveyor);

    // std::string s = std::string("cherry ") + cherry_conveyor.X);
    // qInfo() << "cherry " << cherry_conveyor.X << cherry_conveyor.Y << cherry_conveyor.Type;
  }

  return cherry_array;
}
