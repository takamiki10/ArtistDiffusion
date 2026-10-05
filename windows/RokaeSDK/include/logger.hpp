#pragma once
#include "core.hpp"
#include <atomic>
#include <deque>
#include <mutex>
#include <optional>
#include <ostream>

namespace experiment {
enum Field : uint32_t { MeasuredQ=1, MeasuredDq=2, EndPose=4, TargetQ=8, RobotState=16 };
struct Record {
 uint64_t sequence=0, read_start_ns=0, read_end_ns=0, state_read_start_ns=0, state_read_end_ns=0;
 uint32_t update_bytes=0, reported_valid=0, valid=0;
 int32_t sdk_error=0;
 std::array<int32_t,5> field_rc{{-1,-1,-1,-1,-1}};
 Q q{},dq{},target{};
 std::array<double,16> pose{};
 std::array<int32_t,3> state{{0,0,0}};
 bool has_source_sequence=false; // Only if a source truly exposes it; SDK adapter cannot invent it.
 uint64_t source_sequence=0;
 Record();
};
struct Event { std::string code; uint64_t sequence; std::string detail; uint64_t host_event_ns, read_start_ns, read_end_ns; };
Record validated_record(Record);
std::string encode_record(const Record&);
Record decode_record(const std::string& frame);
class RawParser {
 std::string buffer_;
 public:
 std::vector<Record> feed(const std::string&);
 void finish() const;
};
class BoundedQueue {
 std::mutex mutex_; std::deque<Record> data_; size_t capacity_;
 public:
 explicit BoundedQueue(size_t capacity);
 bool push(const Record&);
 std::optional<Record> pop();
};
class Logger {
 BoundedQueue queue_;
 std::ostream& raw_; std::ostream& event_out_;
 std::mutex event_mutex_;
 std::vector<Event> events_;
 size_t emitted_=0;
 std::atomic<uint64_t> acquired_{0};
 uint64_t last_end_=0;
 bool have_end_=false,have_source_=false;
 uint64_t last_source_=0;
 std::atomic<bool> invalid_{false},writer_failed_{false};
 void event(std::string,uint64_t,std::string,uint64_t read_start_ns=0,uint64_t read_end_ns=0);
 public:
 Logger(size_t,std::ostream&,std::ostream&);
 bool ingest(Record); // Single acquisition owner. Timeouts produce events, never stale samples.
 bool pump();         // Single writer owner; may run on a separate thread.
 void flush();
 bool invalid() const {return invalid_.load();}
 bool writer_failed() const {return writer_failed_.load();}
 std::vector<Event> events(); // Emergency in-memory diagnostics survive sink failure.
};
struct Conversion {
 std::string trial_id;
 uint64_t origin_ns;
 bool joint_mapping_verified=false;
 std::array<int,6> controller_to_jetson{{1,2,3,4,5,6}};
 Q sign{{1,1,1,1,1,1}},offset{};
};
// Diagnostic CSV retains invalid/duplicate rows with empty invalid values.
// It does not assert experiment-valid time alignment.
void csv_header(std::ostream&);
void csv_record(std::ostream&,const Record&,const Conversion&);
}
