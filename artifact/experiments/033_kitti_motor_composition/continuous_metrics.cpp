// Native KITTI metric functions, only on observed connected components.
// Local no-I/O shim for the unused server wrapper; no upstream source edits.
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
  if(!(std::cin>>count) || count<2 || count>100000) return 2;
  std::vector<Matrix> gt,pred;
  std::vector<int> component;
  for(int i=0;i<count;++i) {
    int comp;
    if(!(std::cin>>comp)) return 3;
    component.push_back(comp);
    Matrix g=Matrix::eye(4),p=Matrix::eye(4);
    for(int r=0;r<3;++r) for(int c=0;c<4;++c)
      if(!(std::cin>>g.val[r][c])) return 4;
    for(int r=0;r<3;++r) for(int c=0;c<4;++c)
      if(!(std::cin>>p.val[r][c])) return 5;
    gt.push_back(g);pred.push_back(p);
  }
  auto distance=trajectoryDistances(gt);
  std::cout<<std::setprecision(17);
  for(int first=0;first<count;first+=10) {
    for(int k=0;k<num_lengths;++k) {
      float length=lengths[k];
      int last=lastFrameFromSegmentLength(distance,first,length);
      if(last<0) continue;
      if(component[first]!=component[last]) {
        std::cout<<"F "<<first<<" "<<last<<" "<<length<<"\n";
        continue;
      }
      Matrix truth=Matrix::inv(gt[first])*gt[last];
      Matrix estimate=Matrix::inv(pred[first])*pred[last];
      Matrix error=Matrix::inv(estimate)*truth;
      std::cout<<"S "<<first<<" "<<last<<" "<<length<<" "
               <<translationError(error)/length<<" "<<rotationError(error)/length<<" "
               <<translationError(truth)/length<<" "<<rotationError(truth)/length<<"\n";
    }
  }
  return 0;
}
