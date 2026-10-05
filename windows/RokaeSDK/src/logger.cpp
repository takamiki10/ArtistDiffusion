#include "logger.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace experiment {
Record::Record(){double n=std::numeric_limits<double>::quiet_NaN();
 q.fill(n);
 dq.fill(n);
 target.fill(n);
 pose.fill(n);
 }
namespace {
template<size_t N> bool finite(const std::array<double,N>& a){for(double v:a)if(!std::isfinite(v))return false;
 return true;
 }
bool pose_valid(const std::array<double,16>& p){
 if(!finite(p)||std::abs(p[12])>1e-8||std::abs(p[13])>1e-8||std::abs(p[14])>1e-8||std::abs(p[15]-1)>1e-8)return false;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){double dot=0;
 for(int k=0;k<3;++k)dot+=p[4*i+k]*p[4*j+k];
 if(std::abs(dot-(i==j?1:0))>1e-6)return false;
 }
 double det=p[0]*(p[5]*p[10]-p[6]*p[9])-p[1]*(p[4]*p[10]-p[6]*p[8])+p[2]*(p[4]*p[9]-p[5]*p[8]);
 return std::abs(det-1)<1e-6;
 }
void put(std::string& s,uint64_t v,int n){for(int i=0;i<n;++i)s.push_back(char((v>>(8*i))&255));
 }
uint64_t get(const std::string& s,size_t& pos,int n){if(pos+size_t(n)>s.size())throw std::runtime_error("truncated raw payload");
 uint64_t v=0;
 for(int i=0;i<n;++i)v|=uint64_t(uint8_t(s[pos++]))<<(8*i);
 return v;
 }
void put_double(std::string& s,double v){uint64_t bits;
 std::memcpy(&bits,&v,8);
 put(s,bits,8);
 }
double get_double(const std::string& s,size_t& p){auto bits=get(s,p,8);
 double d;
 std::memcpy(&d,&bits,8);
 return d;
 }
int32_t get_i32(const std::string& s,size_t& p){auto u=uint32_t(get(s,p,4));
 int32_t v;
 std::memcpy(&v,&u,4);
 return v;
 }
constexpr uint32_t payload_size=40+16+20+6*8*3+16*8+3*4+1+8;
 }
Record validated_record(Record r){
 r.valid=0;
 if(r.update_bytes==0||r.sdk_error!=0)return r;
 if((r.reported_valid&MeasuredQ)&&r.field_rc[0]==0&&finite(r.q))r.valid|=MeasuredQ;
 if((r.reported_valid&MeasuredDq)&&r.field_rc[1]==0&&finite(r.dq))r.valid|=MeasuredDq;
 if((r.reported_valid&EndPose)&&r.field_rc[2]==0&&pose_valid(r.pose))r.valid|=EndPose;
 if((r.reported_valid&TargetQ)&&r.field_rc[3]==0&&finite(r.target))r.valid|=TargetQ;
 if((r.reported_valid&RobotState)&&r.field_rc[4]==0&&r.state_read_end_ns>=r.state_read_start_ns)r.valid|=RobotState;
 return r;
 }
std::string encode_record(const Record& r){
 std::string p;
 put(p,r.sequence,8);
 put(p,r.read_start_ns,8);
 put(p,r.read_end_ns,8);
 put(p,r.state_read_start_ns,8);
 put(p,r.state_read_end_ns,8);
 put(p,r.update_bytes,4);
 put(p,r.reported_valid,4);
 put(p,r.valid,4);
 put(p,uint32_t(r.sdk_error),4);
 for(auto v:r.field_rc)put(p,uint32_t(v),4);
 for(double v:r.q)put_double(p,v);
 for(double v:r.dq)put_double(p,v);
 for(double v:r.pose)put_double(p,v);
 for(double v:r.target)put_double(p,v);
 for(auto v:r.state)put(p,uint32_t(v),4);
 put(p,r.has_source_sequence?1:0,1);
 put(p,r.source_sequence,8);
 if(p.size()!=payload_size)throw std::logic_error("record size mismatch");
 std::string out="XCRAW001";
 put(out,p.size(),4);
 out+=p;
 out+=sha256(p);
 return out;
 }
Record decode_record(const std::string& frame){
 if(frame.size()<12||frame.substr(0,8)!="XCRAW001")throw std::runtime_error("unknown raw format");
 size_t x=8;
 auto n=get(frame,x,4);
 if(n!=payload_size||frame.size()!=12+n+64)throw std::runtime_error("incorrect raw size");
 auto p=frame.substr(12,n);
 if(sha256(p)!=frame.substr(12+n,64))throw std::runtime_error("corrupt raw record SHA256");
 size_t pos=0;
 Record r;
 r.sequence=get(p,pos,8);
 r.read_start_ns=get(p,pos,8);
 r.read_end_ns=get(p,pos,8);
 r.state_read_start_ns=get(p,pos,8);
 r.state_read_end_ns=get(p,pos,8);
 r.update_bytes=uint32_t(get(p,pos,4));
 r.reported_valid=uint32_t(get(p,pos,4));
 r.valid=uint32_t(get(p,pos,4));
 r.sdk_error=get_i32(p,pos);
 for(auto& v:r.field_rc)v=get_i32(p,pos);
 for(auto& v:r.q)v=get_double(p,pos);
 for(auto& v:r.dq)v=get_double(p,pos);
 for(auto& v:r.pose)v=get_double(p,pos);
 for(auto& v:r.target)v=get_double(p,pos);
 for(auto& v:r.state)v=get_i32(p,pos);
 auto present=get(p,pos,1);
 if(present>1)throw std::runtime_error("bad source-sequence flag");
 r.has_source_sequence=present!=0;
 r.source_sequence=get(p,pos,8);
 if((r.reported_valid&~31u)||(r.valid&~31u)||r.valid!=validated_record(r).valid)throw std::runtime_error("inconsistent field validity");
 return r;
 }
std::vector<Record> RawParser::feed(const std::string& bytes){buffer_+=bytes;
 std::vector<Record> out;
 while(buffer_.size()>=12){if(buffer_.substr(0,8)!="XCRAW001")throw std::runtime_error("malformed raw framing");
 size_t p=8;
 auto n=get(buffer_,p,4);
 if(n!=payload_size)throw std::runtime_error("unsupported raw size");
 size_t total=12+n+64;
 if(buffer_.size()<total)break;
 out.push_back(decode_record(buffer_.substr(0,total)));
 buffer_.erase(0,total);
 }return out;
 }
void RawParser::finish()const{if(!buffer_.empty())throw std::runtime_error("incomplete raw tail retained; no silent repair");
 }
BoundedQueue::BoundedQueue(size_t n):capacity_(n){if(!n)throw std::invalid_argument("queue capacity must be positive");
 }
bool BoundedQueue::push(const Record& r){std::lock_guard<std::mutex> g(mutex_);
 if(data_.size()>=capacity_)return false;
 data_.push_back(r);
 return true;
 }
std::optional<Record> BoundedQueue::pop(){std::lock_guard<std::mutex> g(mutex_);
 if(data_.empty())return {};
 auto r=data_.front();
 data_.pop_front();
 return r;
 }
Logger::Logger(size_t n,std::ostream& raw,std::ostream& events):queue_(n),raw_(raw),event_out_(events){}
void Logger::event(std::string code,uint64_t seq,std::string detail,uint64_t start,uint64_t end){
 auto now=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
 std::lock_guard<std::mutex> g(event_mutex_);
 events_.push_back({std::move(code),seq,std::move(detail),now,start,end});
 }
std::vector<Event> Logger::events(){std::lock_guard<std::mutex> g(event_mutex_);
 return events_;
 }
bool Logger::ingest(Record r){
 if(writer_failed_)return false;
 if(r.sdk_error!=0){invalid_=true;
 event("sdk_error",acquired_,std::to_string(r.sdk_error),r.read_start_ns,r.read_end_ns);
 }
 if(r.update_bytes==0){event("timeout_no_sample",acquired_,"no new frame; cached values not logged",r.read_start_ns,r.read_end_ns);
 return true;
 }
 r.sequence=acquired_++;
 r=validated_record(r);
 if(r.read_end_ns<r.read_start_ns||(have_end_&&r.read_end_ns<=last_end_)){invalid_=true;
 event("invalid_or_duplicate_host_time",r.sequence,"raw sample retained");
 }have_end_=true;
 last_end_=r.read_end_ns;
 if(!(r.valid&MeasuredQ)){invalid_=true;
 event("missing_or_malformed_measured_q",r.sequence,"raw sample retained; not an analyzable trial");
 }
 if(r.valid!=r.reported_valid){invalid_=true;
 event("field_validation_error",r.sequence,"reported payload/return code inconsistent; native values retained");
 }
 if(r.has_source_sequence){if(have_source_&&r.source_sequence!=last_source_+1){invalid_=true;
 event("source_sequence_gap_or_reset",r.sequence,"only detectable when source counter really exists");
 }have_source_=true;
 last_source_=r.source_sequence;
 }
 if(!queue_.push(r)){invalid_=true;
 event("queue_overflow",r.sequence,"sample lost; acquisition invalid; producer must stop");
 return false;
 }return true;
 }
bool Logger::pump(){
 if(writer_failed_)return false;
 bool did=false;
 try{auto r=queue_.pop();
 if(r){auto bytes=encode_record(*r);
 raw_.write(bytes.data(),std::streamsize(bytes.size()));
 if(!raw_)throw std::runtime_error("raw write failed");
 did=true;
 }
  auto e=events();
 while(emitted_<e.size()){auto& x=e[emitted_];
 event_out_<<"{\"code\":"<<json_escape(x.code)<<",\"sequence\":"<<x.sequence<<",\"host_event_ns\":"<<x.host_event_ns<<",\"read_start_ns\":"<<x.read_start_ns<<",\"read_end_ns\":"<<x.read_end_ns<<",\"detail\":"<<json_escape(x.detail)<<"}\n";
 if(!event_out_)throw std::runtime_error("event write failed");
 ++emitted_;
 did=true;
 }
 }catch(...){invalid_=true;
 writer_failed_=true;
 event("writer_failure",acquired_,"sink failed; stop acquisition; emergency diagnostics remain in memory");
 return false;
 }return did;
 }
void Logger::flush(){while(pump()){}try{raw_.flush();
 event_out_.flush();
 if(!raw_||!event_out_)throw std::runtime_error("flush failed");
 }catch(...){invalid_=true;
 writer_failed_=true;
 event("writer_failure",acquired_,"flush failure");
 }}
void csv_header(std::ostream& o){o<<"trial_id,timestamp,q1_measured,q2_measured,q3_measured,q4_measured,q5_measured,q6_measured,sequence_number,host_read_start_ns,host_read_end_ns,reported_validity,field_validity,sdk_update_bytes,sdk_error,controller_timestamp";
 for(int j=1;j<=6;++j)o<<",q"<<j<<"_velocity_measured";
 for(int j=1;j<=6;++j)o<<",q"<<j<<"_commanded";
 for(int j=0;j<16;++j)o<<",end_pose_matrix_"<<j;
 o<<",robot_status,power_state,operate_mode,state_read_start_ns,state_read_end_ns,source_sequence";
 for(int j=0;j<5;++j)o<<",field_rc_"<<j;
 o<<'\n';
 }
void csv_record(std::ostream& o,const Record& r,const Conversion& c){
 if(!c.joint_mapping_verified)throw std::runtime_error("CSV requires verified joint mapping");
 if(c.trial_id.empty()||c.trial_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)throw std::runtime_error("unsafe trial id");
 auto order=c.controller_to_jetson;
 std::sort(order.begin(),order.end());
 if(order!=std::array<int,6>{{1,2,3,4,5,6}})throw std::runtime_error("mapping not a permutation");
 for(int j=0;j<6;++j)if((c.sign[j]!=1&&c.sign[j]!=-1)||!std::isfinite(c.offset[j]))throw std::runtime_error("invalid sign/offset");
 auto converted=[&](const Q& native,bool velocity){Q q{};
 for(int j=0;j<6;++j)q[c.controller_to_jetson[j]-1]=c.sign[j]*(native[j]-(velocity?0:c.offset[j]));
 return q;
 };
 double rel=r.read_end_ns>=c.origin_ns?double(r.read_end_ns-c.origin_ns)/1e9:-double(c.origin_ns-r.read_end_ns)/1e9;
 o<<c.trial_id<<','<<precise(rel);
 auto values=[&](const auto& a,bool valid){for(auto v:a){o<<',';
 if(valid)o<<precise(v);
 }};
 values(converted(r.q,false),r.valid&MeasuredQ);
 o<<','<<r.sequence<<','<<r.read_start_ns<<','<<r.read_end_ns<<','<<r.reported_valid<<','<<r.valid<<','<<r.update_bytes<<','<<r.sdk_error<<',';
 values(converted(r.dq,true),r.valid&MeasuredDq);
 values(converted(r.target,false),r.valid&TargetQ);
 values(r.pose,r.valid&EndPose);
 for(auto v:r.state){o<<',';
 if(r.valid&RobotState)o<<v;
 }o<<','<<r.state_read_start_ns<<','<<r.state_read_end_ns<<',';
 if(r.has_source_sequence)o<<r.source_sequence;
 for(auto rc:r.field_rc)o<<','<<rc;
 o<<'\n';
 if(!o)throw std::runtime_error("CSV write failed");
 }
}
