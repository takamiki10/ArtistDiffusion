#include "core.hpp"
#include "logger.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace experiment;
 namespace fs=std::filesystem;
 constexpr const char* manifest_pin="dbaba32746457240385aaf123a91f84164cba3a4986337addbf7a0c3d2a6506f";
 std::ofstream output(const fs::path& p){std::ofstream f(p,std::ios::binary);
 f.exceptions(std::ios::failbit|std::ios::badbit);
 return f;
 }
void new_directory(const fs::path& p){if(!fs::create_directory(p))throw std::runtime_error("output directory must be new: "+p.u8string());
 }
double old_round(double x){std::ostringstream o;
 o.imbue(std::locale::classic());
 o<<std::fixed<<std::setprecision(4)<<x;
 return std::stod(o.str());
 }
void audit(const fs::path& package,const fs::path& out){
 Readiness state;
 state.begin_validation();
 auto manifest=verify_package(package,manifest_pin);
 std::vector<Reference> refs;
 for(auto path:{"path_0001","path_0003","path_0006"})for(auto cond:{"A","B"})refs.emplace_back(load_trajectory(package,std::string("trajectories/")+path+"/"+cond+".csv",manifest));
 state.validated();
 new_directory(out);
 auto a=output(out/"command_audit.csv");
 a<<"trajectory,sha256,joint,pl_max_speed_rad_s,interior_fd_acc_rad_s2,c2_max_speed_rad_s,c2_max_acc_rad_s2,c2_max_jerk_rad_s3,rt_max_step_rad,c2_min_rad,c2_max_rad,pl_start_speed_rad_s,pl_end_speed_rad_s\n";
 auto q=output(out/"representation_audit.csv");
 q<<"path,changed_joint_entries,old_erased,new_erased,old_max_serialization_error_rad,new_max_serialization_error_rad,binary_bit_mismatches,text_bit_mismatches,knot_bit_mismatches\n";
 for(size_t n=0;n<refs.size();++n){auto& r=refs[n];
 auto d=demands(r);
 for(int j=0;j<6;++j){a<<r.trajectory().name<<','<<r.trajectory().sha256<<','<<j+1;
 for(auto field:{d.pl_speed,d.interior_fd_acc,d.c2_speed,d.c2_acc,d.c2_jerk,d.rt_step,d.lower,d.upper,d.start_pl_speed,d.end_pl_speed})a<<','<<precise(field[j]);
 a<<'\n';
 }
  auto stream=output(out/(std::string(n<2?"path_0001":n<4?"path_0003":"path_0006")+(n%2?"_B":"_A")+"_reference_1ms.csv"));
 stream<<"nominal_time_seconds,q1,q2,q3,q4,q5,q6\n";
 for(int i=0;i<=10000;++i){double t=double(i)/1000;
 stream<<precise(t);
 for(double v:r.at(t))stream<<','<<precise(v);
 stream<<'\n';
 }
 }
 for(size_t n=0;n<refs.size();n+=2){size_t changed=0,old_erased=0,new_erased=0,binary_bad=0,text_bad=0,knot_bad=0;
 double old_error=0,new_error=0;
 auto& A=refs[n].trajectory();
 auto& B=refs[n+1].trajectory();
 for(size_t i=0;i<100;++i){auto qa=refs[n].at(A.knots[i].t),qb=refs[n+1].at(B.knots[i].t);
 for(int j=0;j<6;++j){double x=A.knots[i].q[j],y=B.knots[i].q[j];
 if(x!=y){++changed;
 if(old_round(x)==old_round(y))++old_erased;
 if(std::stod(precise(x))==std::stod(precise(y)))++new_erased;
 }if(!same_bits(x,qa[j]))++knot_bad;
 if(!same_bits(y,qb[j]))++knot_bad;
 }
   for(auto* t:{&A,&B}){Record raw;
 raw.update_bytes=48;
 raw.reported_valid=MeasuredQ;
 raw.field_rc[0]=0;
 raw.q=t->knots[i].q;
 auto decoded=decode_record(encode_record(validated_record(raw)));
 for(int j=0;j<6;++j){double v=t->knots[i].q[j],back=std::stod(precise(v));
 old_error=std::max(old_error,std::abs(v-old_round(v)));
 new_error=std::max(new_error,std::abs(v-back));
 if(!same_bits(v,back))++text_bad;
 if(!same_bits(v,decoded.q[j]))++binary_bad;
 }}
  }q<<(n==0?"path_0001":n==2?"path_0003":"path_0006")<<','<<changed<<','<<old_erased<<','<<new_erased<<','<<precise(old_error)<<','<<precise(new_error)<<','<<binary_bad<<','<<text_bad<<','<<knot_bad<<'\n';
 if(binary_bad||text_bad||knot_bad||new_erased)throw std::runtime_error("representation/knot preservation failure");
 }
 auto meta=output(out/"audit_metadata.json");
 meta<<"{\"scope\":\"OFFLINE_REFERENCE_ONLY_NO_ROBOT\",\"package_manifest_sha256\":\""<<manifest_pin<<"\",\"verified_files\":"<<manifest.size()<<",\"sample_count\":100,\"spacing_tolerance_s\":1e-12,\"rt_cycle_s\":0.001,\"reference_updates\":10001,\"interpolation\":\"C2 quintic Hermite, centered interior derivatives, zero endpoint v/a\",\"continuous_extrema\":\"derivative-root numerical search, not interval certification\",\"hardware_execution_enabled\":false}\n";
 std::cout<<"Verified "<<manifest.size()<<" package hashes; audited six trajectories. OFFLINE ONLY.\n";
 }
Record synthetic(uint64_t i){Record r;
 r.update_bytes=320;
 r.read_start_ns=1000000000+i*1000000;
 r.read_end_ns=r.read_start_ns+10000;
 r.reported_valid=MeasuredQ|MeasuredDq|EndPose|TargetQ|RobotState;
 r.field_rc.fill(0);
 r.q={0.1+double(i)*1e-8,-0.2,0.3,-0.4,0.5,-0.6};
 r.dq.fill(0);
 r.target=r.q;
 r.pose.fill(0);
 for(int j=0;j<4;++j)r.pose[5*j]=1;
 r.pose[3]=0.123;
 r.pose[7]=0.234;
 r.pose[11]=0.345;
 r.state={0,0,0};
 r.state_read_start_ns=r.read_start_ns;
 r.state_read_end_ns=r.read_end_ns;
 return r;
 }
void demo(const fs::path& out){
 new_directory(out);
 auto raw=output(out/"SYNTHETIC_ONLY.sdk_decoded_raw.bin"),events=output(out/"SYNTHETIC_ONLY.events.jsonl");
 Logger logger(128,raw,events);
 std::atomic<bool> done{false};
 std::thread writer([&]{while(!done){if(!logger.pump())std::this_thread::yield();}logger.flush();});
 for(uint64_t i=0;i<32;++i)if(!logger.ingest(synthetic(i)))break;
 done=true;
 writer.join();
 raw.close();
 events.close();
 auto bytes=read_file(out/"SYNTHETIC_ONLY.sdk_decoded_raw.bin");
 RawParser parser;
 auto records=parser.feed(bytes);
 parser.finish();
 auto csv=output(out/"SYNTHETIC_ONLY_measured_joints.csv");
 csv_header(csv);
 Conversion c{"SYNTHETIC_ONLY",1000000000,true};
 for(auto& r:records)csv_record(csv,r,c);
 auto metadata=output(out/"session.json");
 metadata<<"{\"schema_version\":1,\"scope\":\"SYNTHETIC_TEST_ONLY_NOT_ROBOT_MEASUREMENTS\",\"source\":\"in-process synthetic generator\",\"native_format\":\"XCRAW001 sdk_decoded_raw\",\"joint_units\":\"rad\",\"velocity_units\":\"rad/s\",\"end_pose_frame\":\"synthetic base\",\"orientation\":\"row-major rotation matrix\",\"source_sequence_available\":false,\"controller_timestamp_available\":false,\"timestamp_basis\":\"synthetic host read end minus synthetic capture origin; not controller time\",\"origin_ns\":1000000000,\"mapping\":\"synthetic identity\",\"invalid\":"<<(logger.invalid()?"true":"false")<<",\"record_count\":"<<records.size()<<"}\n";
 if(logger.invalid()||records.size()!=32)throw std::runtime_error("synthetic capture failed");
 std::cout<<"32 SYNTHETIC records written and converted; no robot.\n";
 }
int main(int argc,char** argv){
 try{
  if(argc==2&&std::string(argv[1])=="--stationary-capture"){std::cerr<<"BLOCKED: no hardware backend; safe connection lifecycle and SDK compatibility unverified. No connection attempted.\n";
 return 3;
 }
  if(argc==4&&std::string(argv[1])=="--audit"){audit(fs::u8path(argv[2]),fs::u8path(argv[3]));
 return 0;
 }
  if(argc==3&&std::string(argv[1])=="--synthetic-capture"){demo(fs::u8path(argv[2]));
 return 0;
 }
  std::cerr<<"Offline modes only:\n  --audit PACKAGE NEW_OUTPUT_DIRECTORY\n  --synthetic-capture NEW_OUTPUT_DIRECTORY\n  --stationary-capture (blocked; no connection)\n";
 return 2;
 }catch(const std::exception& e){std::cerr<<"FAILED CLOSED: "<<e.what()<<'\n';
 return 2;
 }
}
