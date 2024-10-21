#pragma once

#ifndef LLOG_HEADER_ONLY
#include "tic_toc.hpp"
namespace llog {
// // define print color
// const std::string RESET = "\033[0m";
// const std::string RESET_RETURN = "\033[0m\n";
// const std::string BLACK = "\033[1;30m";   /* Black */
// const std::string RED = "\033[1;31m";     /* Red */
// const std::string GREEN = "\033[1;32m";   /* Green */
// const std::string YELLOW = "\033[1;33m";  /* Yellow */
// const std::string BLUE = "\033[1;34m";    /* Blue */
// const std::string MAGENTA = "\033[1;35m"; /* Magenta */
// const std::string CYAN = "\033[1;36m";    /* Cyan */
// const std::string WHITE = "\033[1;37m";   /* White */

TicToc::Ptr CreateTimer(const std::string &name);
std::string GetAllTimingStatistics(int print_level = 100);
void PrintAllTimingStatistics(int print_level = 100);
void Reset();
void PrintLog(int print_level = 100);
void SaveLog(const std::string &_save_path);

void InitValueFile(const std::string &file_path);
void RecordValue(const std::string &name, const float &value);
float GetValue(const std::string &name);
std::string FlashValue(const std::string &file_path, const int &precision = 2);

} // namespace llog
#else

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>

class TicToc {
public:
  using Ptr = std::shared_ptr<TicToc>;

  TicToc() { reset(); }

  void reset() {
    t_sum = 0.0;
    count = 0;
    tic();
  }

  void tic() { start = std::chrono::high_resolution_clock::now(); }

  double toc() {
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
  }

  double toc_avg() {
    t_this = toc();
    t_sum += t_this;
    ++count;
    return t_sum / (double)count;
  }

  double toc_sum() {
    t_this = toc();
    t_sum += t_this;
    ++count;
    return t_sum;
  }

  void add(const double &t) {
    t_sum += t;
    ++count;
  }

  double avg() const {
    if (count)
      return t_sum / (double)count;
    else
      return 0;
  }
  double sum() const { return t_sum; }
  int times() const { return count; }

  double this_time() const { return t_this; }

  static double now() {
    return std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  double t_sum, t_this;
  int count;
};

#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>
namespace llog {
static std::vector<std::pair<std::string, TicToc::Ptr>> timers;

TicToc::Ptr CreateTimer(const std::string &name) {
  TicToc::Ptr timer = std::make_shared<TicToc>();
  timers.emplace_back(name, timer);
  return timer;
}

std::string GetAllTimingStatistics(int print_level = 100) {
  std::string str;
  str += "Timing Statistics [total time/count = avg. time, this time (ms)]:\n";
  for (const auto &it : timers) {
    if (it.first.find_first_not_of(' ') < print_level) {
      str += it.first + ":\t" + std::to_string(it.second->sum() * 1000.0) +
             "/" + std::to_string(it.second->times()) + " = " +
             std::to_string(it.second->avg() * 1000.0) + ", " +
             std::to_string(it.second->this_time() * 1000.0) + "\n";
    }
  }
  return str;
}

void PrintAllTimingStatistics(int print_level = 100) {
  // // Clear Print Time Statistics last time
  // printf("\033[1;1H");
  printf(
      "\nTiming Statistics [total time/count = avg. time, this time (ms)]:\n");
  for (const auto &it : timers) {
    if (it.first.find_first_not_of(' ') < print_level) {
      printf("%s:\t%.3f/%d = %.3f, %.3f\n", it.first.c_str(),
             it.second->sum() * 1000.0, it.second->times(),
             it.second->avg() * 1000.0, it.second->this_time() * 1000.0);
    }
  }
}

void Reset() {
  for (auto &timer : timers) {
    timer.second->reset();
  }
}

void PrintLog(int print_level = 100) { PrintAllTimingStatistics(print_level); }

void SaveLog(const std::string &_save_path) {
  std::ofstream timing_file(_save_path);
  timing_file << llog::GetAllTimingStatistics();
  timing_file.close();
  // printf("%sTiming Statistics saved to %s\n%s", GREEN.c_str(),
  //        _save_path.c_str(), RESET.c_str());
  printf("\033[1;32mTiming Statistics saved to %s\033[0m\n",
         _save_path.c_str());
}

static std::unordered_map<std::string, float> value_map;
static std::ofstream value_file;

void InitValueFile(const std::string &file_path) {
  if (value_file.is_open()) {
    value_file.close();
  }
  value_file = std::ofstream(file_path);
  value_map.clear();
  std::ofstream loss_file(file_path);
  loss_file.close();
}

void RecordValue(const std::string &name, const float &value) {
  value_map[name] = value;
}
float GetValue(const std::string &name) { return value_map[name]; }
std::string FlashValue(const std::string &file_path, const int &precision = 2) {
  if (!value_file.is_open()) {
    value_file = std::ofstream(file_path);

    for (const auto &value : value_map) {
      value_file << value.first << "\t";
    }
    value_file << "\n";
  }

  std::stringstream out_string;
  out_string << std::setprecision(precision);
  for (const auto &value : value_map) {
    value_file << value.second << "\t";
    out_string << ", " << value.first << ": " << value.second;
  }
  value_file << "\n";
  return out_string.str();
}
} // namespace llog
#endif