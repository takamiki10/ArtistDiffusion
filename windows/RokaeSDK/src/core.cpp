#include "core.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace experiment {
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,"IEEE binary64 required");
 std::string read_file(const std::filesystem::path& p) {
 std::ifstream f(p,std::ios::binary);
 if(!f) throw std::runtime_error("cannot read "+p.u8string());
 std::ostringstream b;
 b<<f.rdbuf();
 if(f.bad()) throw std::runtime_error("file read error");
 return b.str();
 }
std::string precise(double v) { std::ostringstream s;
 s.imbue(std::locale::classic());
 s<<std::setprecision(std::numeric_limits<double>::max_digits10)<<v;
 return s.str();
 }
bool same_bits(double a,double b) { return std::memcmp(&a,&b,8)==0;
 }
std::string json_escape(const std::string& s) {
 std::ostringstream o;
 o<<'"';
 for(unsigned char c:s) { if(c=='"'||c=='\\') o<<'\\'<<c;
 else if(c<32) o<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c)<<std::dec;
 else o<<c;
 } o<<'"';
 return o.str();
 }
namespace {
uint32_t rr(uint32_t x,int n) { return (x>>n)|(x<<(32-n));
 }
constexpr uint32_t K[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
 }
std::string sha256(const std::string& input) {
 std::vector<uint8_t> d(input.begin(),input.end());
 uint64_t bits=uint64_t(d.size())*8;
 d.push_back(128);
 while(d.size()%64!=56) d.push_back(0);
 for(int i=7;i>=0;--i) d.push_back(uint8_t(bits>>(i*8)));
 std::array<uint32_t,8> h={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
 for(size_t off=0;off<d.size();off+=64) {
  uint32_t w[64];
 for(int i=0;i<16;++i) w[i]=(uint32_t(d[off+4*i])<<24)|(uint32_t(d[off+4*i+1])<<16)|(uint32_t(d[off+4*i+2])<<8)|d[off+4*i+3];
 for(int i=16;i<64;++i) w[i]=w[i-16]+(rr(w[i-15],7)^rr(w[i-15],18)^(w[i-15]>>3))+w[i-7]+(rr(w[i-2],17)^rr(w[i-2],19)^(w[i-2]>>10));
 auto a=h[0],b=h[1],c=h[2],e=h[4],f=h[5],g=h[6],v=h[7],dd=h[3];
 for(int i=0;i<64;++i) { auto t1=v+(rr(e,6)^rr(e,11)^rr(e,25))+((e&f)^((~e)&g))+K[i]+w[i];
 auto t2=(rr(a,2)^rr(a,13)^rr(a,22))+((a&b)^(a&c)^(b&c));
 v=g;
 g=f;
 f=e;
 e=dd+t1;
 dd=c;
 c=b;
 b=a;
 a=t1+t2;
 }
  h[0]+=a;
 h[1]+=b;
 h[2]+=c;
 h[3]+=dd;
 h[4]+=e;
 h[5]+=f;
 h[6]+=g;
 h[7]+=v;
 }
 std::ostringstream s;
 for(auto x:h) s<<std::hex<<std::setw(8)<<std::setfill('0')<<x;
 return s.str();
 }
std::map<std::string,std::string> parse_manifest(const std::string& s) {
 size_t i=0;
 auto ws=[&]{while(i<s.size()&&(s[i]==' '||s[i]=='\n'||s[i]=='\r'||s[i]=='\t')) ++i;
 };
 auto token=[&](char c){ws();
 if(i==s.size()||s[i++]!=c) throw std::runtime_error("malformed manifest");
 };
 auto str=[&](){ ws();
 if(i==s.size()||s[i++]!='"') throw std::runtime_error("manifest string expected");
 std::string r;
 while(i<s.size()&&s[i]!='"'){char c=s[i++];
 if(static_cast<unsigned char>(c)<32)throw std::runtime_error("manifest control char");
 if(c=='\\'){if(i==s.size())throw std::runtime_error("manifest escape");
 c=s[i++];
 if(c!='\\'&&c!='/'&&c!='"')throw std::runtime_error("unsupported manifest escape");
 }r+=c;
 }if(i==s.size())throw std::runtime_error("unterminated manifest");
 ++i;
 return r;
 };
 std::map<std::string,std::string> out;
 token('{');
 ws();
 if(i<s.size()&&s[i]=='}')throw std::runtime_error("empty manifest");
 while(true){auto key=str();
 token(':');
 auto value=str();
 if(value.size()!=64||value.find_first_not_of("0123456789abcdef")!=std::string::npos)throw std::runtime_error("invalid SHA256");
 if(!out.emplace(key,value).second)throw std::runtime_error("duplicate manifest entry");
 ws();
 if(i<s.size()&&s[i]=='}'){++i;
 break;
 }token(',');
 }
 ws();
 if(i!=s.size())throw std::runtime_error("manifest trailing data");
 return out;
 }
std::map<std::string,std::string> verify_package(const std::filesystem::path& root,const std::string& trusted) {
 auto bytes=read_file(root/"SHA256SUMS.json");
 if(sha256(bytes)!=trusted)throw std::runtime_error("untrusted package manifest SHA256");
 auto m=parse_manifest(bytes);
 auto base=std::filesystem::canonical(root);
 for(auto& entry:m){auto rel=std::filesystem::u8path(entry.first);
 if(rel.is_absolute()||entry.first.find('\\')!=std::string::npos||entry.first.find(':')!=std::string::npos)throw std::runtime_error("unsafe manifest path");
 for(auto& part:rel)if(part=="..")throw std::runtime_error("manifest traversal");
 auto p=std::filesystem::canonical(base/rel);
 auto check=p.lexically_relative(base);
 if(check.empty()||*check.begin()=="..")throw std::runtime_error("manifest escaped package");
 if(sha256(read_file(p))!=entry.second)throw std::runtime_error("package hash mismatch: "+entry.first);
 }
 return m;
 }
Trajectory parse_csv(const std::string& input,const std::string& name){
 std::string b=input;
 if(b.rfind("\xef\xbb\xbf",0)==0)b.erase(0,3);
 std::istringstream f(b);
 std::string line;
 auto trim_cr=[](std::string& x){if(!x.empty()&&x.back()=='\r')x.pop_back();
 };
 if(!std::getline(f,line))throw std::runtime_error("empty CSV");
 trim_cr(line);
 if(line!="time_seconds,q1,q2,q3,q4,q5,q6")throw std::runtime_error("incorrect CSV header");
 Trajectory t{};
 t.name=name;
 t.sha256=sha256(input);
 size_t count=0;
 while(std::getline(f,line)){trim_cr(line);
 if(count>=100)throw std::runtime_error("extra CSV rows");
 std::array<double,7> values{};
 size_t from=0;
 for(size_t j=0;j<7;++j){size_t end=line.find(',',from);
 if((j<6&&end==std::string::npos)||(j==6&&end!=std::string::npos))throw std::runtime_error("CSV field count");
 if(end==std::string::npos)end=line.size();
 auto first=line.data()+from;
 auto last=line.data()+end;
 auto parsed=std::from_chars(first,last,values[j],std::chars_format::general);
 if(first==last||parsed.ec!=std::errc{}||parsed.ptr!=last||!std::isfinite(values[j]))throw std::runtime_error("malformed/nonfinite CSV number");
 from=end+1;
 }
  t.knots[count].t=values[0];
 std::copy(values.begin()+1,values.end(),t.knots[count].q.begin());
 if(count&&values[0]<=t.knots[count-1].t)throw std::runtime_error("nonmonotonic timestamps");
 if(std::abs(values[0]-double(count)*10.0/99.0)>spacing_tolerance_s)throw std::runtime_error("unexpected timestamp grid");
 if(count&&std::abs(values[0]-t.knots[count-1].t-10.0/99.0)>spacing_tolerance_s)throw std::runtime_error("incorrect timestamp spacing");
 ++count;
 }
 if(count!=100||t.knots[0].t!=0.0||t.knots[99].t!=10.0)throw std::runtime_error("requires 100 rows and exact endpoints 0,10");
 return t;
 }
Trajectory load_trajectory(const std::filesystem::path& root,const std::string& name,const std::map<std::string,std::string>& m){auto it=m.find(name);
 if(it==m.end())throw std::runtime_error("trajectory absent from verified manifest");
 auto b=read_file(root/std::filesystem::u8path(name));
 if(sha256(b)!=it->second)throw std::runtime_error("trajectory changed since verification");
 return parse_csv(b,name);
 }
double eval(const Poly& c,double u){double v=0;
 for(auto i=c.rbegin();i!=c.rend();++i)v=v*u+*i;
 return v;
 }
Poly derivative(const Poly& c){Poly p;
 for(size_t i=1;i<c.size();++i)p.push_back(double(i)*c[i]);
 if(p.empty())p.push_back(0);
 return p;
 }
std::vector<double> roots_unit(const Poly& input){
 Poly p=input;
 while(p.size()>1&&p.back()==0)p.pop_back();
 if(p.size()<2)return {};
 if(p.size()==2){double r=-p[0]/p[1];
 return r>=0&&r<=1?std::vector<double>{r}:std::vector<double>{};
 }
 auto critical=roots_unit(derivative(p));
 critical.push_back(0);
 critical.push_back(1);
 std::sort(critical.begin(),critical.end());
 std::vector<double> roots;
 double scale=1;
 for(double c:p)scale+=std::abs(c);
 for(double r:critical)if(std::abs(eval(p,r))<=1e-12*scale)roots.push_back(r);
 for(size_t i=1;i<critical.size();++i){double l=critical[i-1],r=critical[i],fl=eval(p,l),fr=eval(p,r);
 if((fl<0&&fr>0)||(fl>0&&fr<0)){for(int k=0;k<64;++k){double mid=(l+r)/2,fm=eval(p,mid);
 if((fl<0&&fm<0)||(fl>0&&fm>0)){l=mid;
 fl=fm;
 }else r=mid;
 }roots.push_back((l+r)/2);
 }}
 std::sort(roots.begin(),roots.end());
 roots.erase(std::unique(roots.begin(),roots.end(),[](double a,double b){return std::abs(a-b)<1e-10;}),roots.end());
 return roots;
 }
Reference::Reference(Trajectory t):traj_(std::move(t)){
 std::array<Q,100> v{},a{};
 std::array<Q,99> slope{};
 std::array<double,99> h{};
 for(size_t i=0;i<99;++i){h[i]=traj_.knots[i+1].t-traj_.knots[i].t;
 for(int j=0;j<6;++j)slope[i][j]=(traj_.knots[i+1].q[j]-traj_.knots[i].q[j])/h[i];
 }
 for(size_t i=1;i<99;++i)for(int j=0;j<6;++j){v[i][j]=(h[i]*slope[i-1][j]+h[i-1]*slope[i][j])/(h[i-1]+h[i]);
 a[i][j]=2*(slope[i][j]-slope[i-1][j])/(h[i-1]+h[i]);
 }
 for(size_t i=0;i<99;++i){Segment s{};
 s.t=traj_.knots[i].t;
 s.h=h[i];
 for(int j=0;j<6;++j){double c0=traj_.knots[i].q[j],c1=h[i]*v[i][j],c2=h[i]*h[i]*a[i][j]/2;
 double A=traj_.knots[i+1].q[j]-c0-c1-c2,B=h[i]*v[i+1][j]-c1-2*c2,C=h[i]*h[i]*a[i+1][j]-2*c2;
 s.c[j]={c0,c1,c2,10*A-4*B+C/2,-15*A+7*B-C,6*A-3*B+C/2};
 }segments_.push_back(s);
 }
}
Q Reference::at(double t)const{if(!std::isfinite(t))throw std::runtime_error("nonfinite evaluation time");
 auto& k=traj_.knots;
 if(t<=0)return k.front().q;
 if(t>=10)return k.back().q;
 auto it=std::lower_bound(k.begin(),k.end(),t,[](const Knot& x,double y){return x.t<y;});
 if(it->t==t)return it->q;
 auto index=size_t(it-k.begin()-1);
 auto& s=segments_[index];
 Q q;
 for(int j=0;j<6;++j)q[j]=eval(s.c[j],(t-s.t)/s.h);
 return q;
 }
Q Reference::linear_at(double t)const{if(!std::isfinite(t))throw std::runtime_error("nonfinite evaluation time");
 auto& k=traj_.knots;
 if(t<=0)return k.front().q;
 if(t>=10)return k.back().q;
 auto it=std::lower_bound(k.begin(),k.end(),t,[](const Knot& x,double y){return x.t<y;});
 if(it->t==t)return it->q;
 auto& l=*(it-1);
 double a=(t-l.t)/(it->t-l.t);
 Q q;
 for(int j=0;j<6;++j)q[j]=l.q[j]+a*(it->q[j]-l.q[j]);
 return q;
 }
Demand demands(const Reference& r){
 Demand d{};
 d.lower.fill(std::numeric_limits<double>::infinity());
 d.upper.fill(-std::numeric_limits<double>::infinity());
 auto& k=r.trajectory().knots;
 for(size_t i=0;i<99;++i)for(int j=0;j<6;++j){double h=k[i+1].t-k[i].t,s=(k[i+1].q[j]-k[i].q[j])/h;
 d.pl_speed[j]=std::max(d.pl_speed[j],std::abs(s));
 if(i==0)d.start_pl_speed[j]=s;
 if(i==98)d.end_pl_speed[j]=s;
 if(i){double hp=k[i].t-k[i-1].t,sp=(k[i].q[j]-k[i-1].q[j])/hp;
 d.interior_fd_acc[j]=std::max(d.interior_fd_acc[j],std::abs(2*(s-sp)/(h+hp)));
 }}
 for(auto& s:r.segments())for(int j=0;j<6;++j){Poly p=s.c[j];
 auto where=roots_unit(derivative(p));
 where.push_back(0);
 where.push_back(1);
 for(double u:where){double q=eval(p,u);
 d.lower[j]=std::min(d.lower[j],q);
 d.upper[j]=std::max(d.upper[j],q);
 }for(int order=1;order<=3;++order){p=derivative(p);
 auto roots=roots_unit(derivative(p));
 roots.push_back(0);
 roots.push_back(1);
 double maximum=0;
 for(double u:roots)maximum=std::max(maximum,std::abs(eval(p,u))/std::pow(s.h,order));
 auto& field=order==1?d.c2_speed:(order==2?d.c2_acc:d.c2_jerk);
 field[j]=std::max(field[j],maximum);
 }}
 Q prev=r.at(0);
 for(int i=1;i<=10000;++i){auto q=r.at(double(i)/1000);
 for(int j=0;j<6;++j)d.rt_step[j]=std::max(d.rt_step[j],std::abs(q[j]-prev[j]));
 prev=q;
 }return d;
 }
}
