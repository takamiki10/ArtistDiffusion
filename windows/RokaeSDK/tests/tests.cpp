#include "core.hpp"
#include "logger.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <fstream>
using namespace experiment;
int assertions=0;
void check(bool v,const char* what){++assertions;if(!v)throw std::runtime_error(what);}
template<class F> void rejects(F f,const char* what){bool bad=false;try{f();}catch(const std::exception&){bad=true;}check(bad,what);}
std::string fixture(){std::ostringstream o;o<<"time_seconds,q1,q2,q3,q4,q5,q6\n";for(int i=0;i<100;++i)o<<precise(i*10.0/99)<<','<<precise(std::sin(i*0.1))<<",-0,0,0,0,0\n";return o.str();}
Record sample(){Record r;r.update_bytes=48;r.read_start_ns=10;r.read_end_ns=20;r.reported_valid=MeasuredQ;r.field_rc[0]=0;r.q={0,-0.0,1e-16,1.2345678901234567,-2.3,3.1};return r;}
bool has(Logger& l,const std::string& code){for(auto& e:l.events())if(e.code==code)return true;return false;}
class FailingBuf:public std::streambuf{std::streamsize xsputn(const char*,std::streamsize)override{return 0;}int overflow(int)override{return traits_type::eof();}};
int main(){try{
 check(sha256("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA empty");check(sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA abc");check(sha256(std::string(1000000,'a'))=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","SHA multiblock");
 auto csv=fixture();auto t=parse_csv(csv,"synthetic");check(same_bits(t.knots[1].q[1],-0.0),"negative zero");
 rejects([&]{parse_csv(csv+"\n","extra");},"blank extra row");rejects([&]{parse_csv(csv+"10,0,0,0,0,0,0\n","extra");},"extra row");rejects([&]{parse_csv(csv.substr(0,csv.rfind('\n',csv.size()-2)+1),"missing");},"missing row");
 for(auto value:{"nan","inf","0junk",""," 0","1e999"}){auto b=csv;auto pos=b.find("\n0,")+3;b.replace(pos,b.find(',',pos)-pos,value);rejects([&]{parse_csv(b,"bad number");},"invalid number");}
 auto header_bad=csv;header_bad.replace(0,12,"Timestamp");rejects([&]{parse_csv(header_bad,"header");},"bad header");
 auto duplicate=csv;auto start=duplicate.find('\n')+1;auto next=duplicate.find('\n',start)+1;duplicate.replace(next,duplicate.find(',',next)-next,"0");rejects([&]{parse_csv(duplicate,"time");},"duplicate time");
 auto m="{\"file\":\""+sha256("abc")+"\"}";check(parse_manifest(m).size()==1,"manifest parse");rejects([&]{parse_manifest(m+"x");},"manifest trailing");rejects([&]{parse_manifest("{\"a\":\"bad\"}");},"manifest hash format");
 {namespace fs=std::filesystem;auto root=fs::path("build_offline")/("manifest_fixture_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directories(root);
 auto write=[&](const char* name,const std::string& bytes){std::ofstream f(root/name,std::ios::binary);f<<bytes;};
 write("file","abc");write("SHA256SUMS.json",m);check(verify_package(root,sha256(m)).size()==1,"whole package verified");
 rejects([&]{verify_package(root,sha256("different"));},"manifest pin mismatch");write("file","abx");rejects([&]{verify_package(root,sha256(m));},"changed package file");
 auto traversal="{\"../outside\":\""+sha256("abc")+"\"}";write("SHA256SUMS.json",traversal);rejects([&]{verify_package(root,sha256(traversal));},"manifest traversal rejected");
 rejects([&]{parse_manifest("{\"a\":\""+sha256("a")+"\",\"a\":\""+sha256("a")+"\"}");},"duplicate manifest path");}
 Reference r(t);for(auto& k:t.knots){auto q=r.at(k.t);for(int j=0;j<6;++j)check(same_bits(q[j],k.q[j]),"exact knots");}
 check(r.at(-1)==t.knots.front().q&&r.at(11)==t.knots.back().q,"outside holds");rejects([&]{r.at(std::numeric_limits<double>::quiet_NaN());},"bad query time");
 auto& segments=r.segments();for(size_t i=1;i<segments.size();++i)for(int j=0;j<6;++j)for(int order=0;order<=2;++order){auto l=segments[i-1].c[j],right=segments[i].c[j];for(int k=0;k<order;++k){l=derivative(l);right=derivative(right);}check(std::abs(eval(l,1)/std::pow(segments[i-1].h,order)-eval(right,0)/std::pow(segments[i].h,order))<1e-8,"C2 continuity");}
 for(int order=1;order<=2;++order)for(auto index:{size_t(0),size_t(98)})for(int j=0;j<6;++j){auto p=segments[index].c[j];for(int k=0;k<order;++k)p=derivative(p);check(std::abs(eval(p,index==0?0:1)/std::pow(segments[index].h,order))<1e-8,"endpoint derivatives");}
 auto demand=demands(r);for(int j=0;j<6;++j)check(demand.rt_step[j]<=demand.c2_speed[j]*cycle_s+1e-12,"step bound");auto roots=roots_unit({0.25,-1,1});check(roots.size()==1&&std::abs(roots[0]-0.5)<1e-10,"double polynomial root");
 auto native=validated_record(sample());auto bytes=encode_record(native);auto back=decode_record(bytes);for(int j=0;j<6;++j)check(same_bits(native.q[j],back.q[j]),"raw binary64 fidelity");
 for(size_t split=0;split<=bytes.size();++split){RawParser p;auto a=p.feed(bytes.substr(0,split));auto b=p.feed(bytes.substr(split)+bytes);p.finish();check(a.size()+b.size()==2,"split/coalesced framing");}
 auto corrupt=bytes;corrupt[20]^=1;rejects([&]{decode_record(corrupt);},"checksum corruption");RawParser incomplete;incomplete.feed(bytes.substr(0,22));rejects([&]{incomplete.finish();},"truncated packet");RawParser wrong;rejects([&]{wrong.feed(std::string(12,'X'));},"bad framing");
 {std::ostringstream raw,events;Logger l(8,raw,events);check(l.ingest(sample()),"valid submit");Record timeout;timeout.q.fill(123);check(l.ingest(timeout),"timeout returns");l.flush();RawParser p;check(p.feed(raw.str()).size()==1,"timeout does not create stale sample");check(has(l,"timeout_no_sample"),"timeout event");check(!l.invalid(),"optional unavailable fields allowed");}
 {std::ostringstream raw,events;Logger l(8,raw,events);auto s=sample();s.field_rc[0]=-1;l.ingest(s);l.flush();check(l.invalid()&&has(l,"missing_or_malformed_measured_q"),"required unavailable");auto decoded=decode_record(raw.str());check(!(decoded.valid&MeasuredQ)&&same_bits(decoded.q[2],s.q[2]),"retain invalid raw values");}
 {std::ostringstream raw,events;Logger l(8,raw,events);auto s=sample();s.q[0]=std::numeric_limits<double>::quiet_NaN();l.ingest(s);l.flush();check(l.invalid()&&std::isnan(decode_record(raw.str()).q[0]),"malformed measured data retained");}
 {auto s=sample();s.reported_valid|=EndPose;s.field_rc[2]=0;s.pose.fill(0);check(!(validated_record(s).valid&EndPose),"invalid transform");}
 {std::ostringstream raw,events;Logger l(1,raw,events);l.ingest(sample());check(!l.ingest(sample()),"overflow rejects sample");l.flush();check(l.invalid()&&has(l,"queue_overflow"),"overflow visible");}
 {std::ostringstream raw,events;Logger l(8,raw,events);l.ingest(sample());l.ingest(sample());l.flush();RawParser p;check(p.feed(raw.str()).size()==2&&has(l,"invalid_or_duplicate_host_time"),"duplicate times retained");}
 {std::ostringstream raw,events;Logger l(8,raw,events);auto s=sample();s.has_source_sequence=true;s.source_sequence=1;l.ingest(s);s.source_sequence=3;s.read_start_ns=30;s.read_end_ns=40;l.ingest(s);l.flush();check(has(l,"source_sequence_gap_or_reset"),"synthetic dropped source frame");}
 {FailingBuf fail;std::ostream raw(&fail);std::ostringstream events;Logger l(8,raw,events);l.ingest(sample());l.flush();check(l.writer_failed()&&l.invalid()&&has(l,"writer_failure"),"raw writer failure");check(!l.ingest(sample()),"stop acquiring after writer failure");}
 {std::ostringstream raw;FailingBuf fail;std::ostream events(&fail);Logger l(8,raw,events);l.ingest(Record{});l.flush();check(l.writer_failed(),"event writer failure");}
 {std::ostringstream o;Conversion c{"test",20,true};csv_header(o);csv_record(o,native,c);check(o.str().find("q1_measured")!=std::string::npos,"CSV required fields");c.joint_mapping_verified=false;rejects([&]{csv_record(o,native,c);},"unknown mapping rejected");}
 {BoundedQueue queue(1024);std::atomic<int> count{0};std::thread producer([&]{for(int i=0;i<1000;++i){auto s=sample();s.sequence=i;while(!queue.push(s))std::this_thread::yield();}});std::thread consumer([&]{while(count<1000){auto v=queue.pop();if(v){if(v->sequence!=uint64_t(count))std::terminate();++count;}else std::this_thread::yield();}});producer.join();consumer.join();check(count==1000,"threaded queue order");}
 Readiness ready;rejects([&]{ready.validated();},"invalid state transition");ready.begin_validation();ready.validated();rejects([&]{ready.execute();},"even READY cannot execute");ready.fault();rejects([&]{ready.execute();},"fault cannot execute");
 std::cout<<"PASS "<<assertions<<" assertions; all inputs synthetic; no robot connection\n";return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL after "<<assertions<<": "<<e.what()<<'\n';return 1;}}
