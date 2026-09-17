// Local I/O adapter for the separately retained GPL-2.0-or-later LIBVISO2.
// No reference poses are accepted. Failed estimates are not extrapolated.
#include "viso_stereo.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc,char **argv) {
  if(argc!=5) return 2;
  VisualOdometryStereo::parameters parameters;
  parameters.calib.f=std::stod(argv[1]);
  parameters.calib.cu=std::stod(argv[2]);
  parameters.calib.cv=std::stod(argv[3]);
  parameters.base=std::stod(argv[4]);
  VisualOdometryStereo viso(parameters);
  std::cout<<std::setprecision(17);
  int frame=0;
  while(true) {
    uint32_t header[2];
    std::cin.read(reinterpret_cast<char*>(header),sizeof(header));
    if(std::cin.gcount()==0 && std::cin.eof()) break;
    if(std::cin.gcount()!=sizeof(header)) return 3;
    const uint32_t width=header[0],height=header[1];
    if(width==0 || height==0 || width>4096 || height>4096) return 4;
    const size_t size=static_cast<size_t>(width)*height;
    std::vector<uint8_t> left(size),right(size);
    std::cin.read(reinterpret_cast<char*>(left.data()),size);
    if(static_cast<size_t>(std::cin.gcount())!=size) return 5;
    std::cin.read(reinterpret_cast<char*>(right.data()),size);
    if(static_cast<size_t>(std::cin.gcount())!=size) return 6;
    int32_t dims[]={static_cast<int32_t>(width),static_cast<int32_t>(height),static_cast<int32_t>(width)};
    const auto begin=std::chrono::steady_clock::now();
    bool ok=viso.process(left.data(),right.data(),dims,false);
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<frame<<" "<<ok<<" "<<viso.getNumberOfMatches()<<" "
             <<viso.getNumberOfInliers()<<" "<<seconds;
    if(ok) {
      Matrix motion=viso.getMotion();
      for(int i=0;i<3;++i) for(int j=0;j<4;++j)
        std::cout<<" "<<motion.val[i][j];
    }
    std::cout<<std::endl;
    ++frame;
  }
  return 0;
}
