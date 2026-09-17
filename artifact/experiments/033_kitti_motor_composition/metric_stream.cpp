// Local no-I/O adapter for the official devkit's unused server wrapper.
// Original geometry and metric functions are included without modification.
#include <string>
#define MAIL_H
class Mail {
 public:
  explicit Mail(std::string = "") {}
  void msg(const char *, ...) {}
  template<class... Args> void finalize(Args...) {}
};
#define main unused_official_main
#include "evaluate_odometry.cpp"
#undef main
#include <iomanip>

int main() {
  int count;
  if(!(std::cin>>count) || count<2 || count>401) return 2;
  std::vector<Matrix> gt,pred;
  for(int i=0;i<count;++i) {
    Matrix g=Matrix::eye(4),p=Matrix::eye(4);
    for(int r=0;r<3;++r) for(int c=0;c<4;++c)
      if(!(std::cin>>g.val[r][c])) return 3;
    for(int r=0;r<3;++r) for(int c=0;c<4;++c)
      if(!(std::cin>>p.val[r][c])) return 4;
    gt.push_back(g);pred.push_back(p);
  }
  auto errors=calcSequenceErrors(gt,pred);
  std::cout<<std::setprecision(17);
  for(const auto &e:errors)
    std::cout<<e.first_frame<<" "<<e.len<<" "<<e.t_err<<" "<<e.r_err<<"\n";
  return 0;
}
