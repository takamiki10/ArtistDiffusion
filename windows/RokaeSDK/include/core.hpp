#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace experiment {
using Q = std::array<double,6>;
struct Knot { double t; Q q; };
struct Trajectory { std::array<Knot,100> knots; std::string sha256; std::string name; };
constexpr double spacing_tolerance_s = 1e-12;
constexpr double cycle_s = 0.001;
std::string sha256(const std::string& bytes);
std::string read_file(const std::filesystem::path&);
std::map<std::string,std::string> parse_manifest(const std::string&);
std::map<std::string,std::string> verify_package(const std::filesystem::path&, const std::string& trusted_manifest_sha);
Trajectory parse_csv(const std::string& bytes, const std::string& name);
Trajectory load_trajectory(const std::filesystem::path&, const std::string&, const std::map<std::string,std::string>&);
std::string precise(double);
std::string json_escape(const std::string&);
bool same_bits(double,double);
using Poly = std::vector<double>;
double eval(const Poly&,double);
Poly derivative(const Poly&);
std::vector<double> roots_unit(const Poly&);
struct Segment { double t,h; std::array<Poly,6> c; };
class Reference {
 public:
  explicit Reference(Trajectory);
  Q at(double t) const;
  Q linear_at(double t) const; // Audit reference only, not selected motion convention.
  const Trajectory& trajectory() const { return traj_; }
  const std::vector<Segment>& segments() const { return segments_; }
 private:
  Trajectory traj_;
  std::vector<Segment> segments_;
};
struct Demand {
  Q pl_speed{}, interior_fd_acc{}, c2_speed{}, c2_acc{}, c2_jerk{}, rt_step{}, lower{}, upper{}, start_pl_speed{}, end_pl_speed{};
};
Demand demands(const Reference&);
enum class OfflineState { Load, Validate, Ready, Fault };
class Readiness {
 public:
  OfflineState state=OfflineState::Load;
  void begin_validation() { state=OfflineState::Validate; }
  void validated() { if(state!=OfflineState::Validate) throw std::logic_error("invalid transition"); state=OfflineState::Ready; }
  void fault() { state=OfflineState::Fault; }
  void execute() const { throw std::logic_error("motion is not implemented in offline build"); }
};
}
